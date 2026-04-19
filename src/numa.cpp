/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "numa.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#if defined(__linux__) && !defined(__ANDROID__)
    #include <sched.h>
#endif

namespace Stockfish {

CpuIndex get_hardware_concurrency() {
    CpuIndex concurrency = std::thread::hardware_concurrency();

    // Get all processors across all processor groups on windows, since
    // hardware_concurrency() only returns the number of processors in
    // the first group, because only these are available to std::thread.
#ifdef _WIN64
    concurrency = std::max<CpuIndex>(concurrency, GetActiveProcessorCount(ALL_PROCESSOR_GROUPS));
#endif

    return concurrency;
}

const CpuIndex SYSTEM_THREADS_NB = std::max<CpuIndex>(1, get_hardware_concurrency());

#if defined(_WIN64)

std::optional<std::set<CpuIndex>> WindowsAffinity::get_combined() const {
    if (!oldApi.has_value())
        return newApi;
    if (!newApi.has_value())
        return oldApi;

    std::set<CpuIndex> intersect;
    std::set_intersection(oldApi->begin(), oldApi->end(), newApi->begin(), newApi->end(),
                          std::inserter(intersect, intersect.begin()));
    return intersect;
}

std::pair<BOOL, std::vector<USHORT>> get_process_group_affinity() {

    // GetProcessGroupAffinity requires the GroupArray argument to be
    // aligned to 4 bytes instead of just 2.
    static constexpr size_t GroupArrayMinimumAlignment = 4;
    static_assert(GroupArrayMinimumAlignment >= alignof(USHORT));

    // The function should succeed the second time, but it may fail if the group
    // affinity has changed between GetProcessGroupAffinity calls. In such case
    // we consider this a hard error, as we Cannot work with unstable affinities
    // anyway.
    static constexpr int MAX_TRIES  = 2;
    USHORT               GroupCount = 1;
    for (int i = 0; i < MAX_TRIES; ++i)
    {
        auto GroupArray = std::make_unique<USHORT[]>(
          GroupCount + (GroupArrayMinimumAlignment / alignof(USHORT) - 1));

        USHORT* GroupArrayAligned = align_ptr_up<GroupArrayMinimumAlignment>(GroupArray.get());

        const BOOL status =
          GetProcessGroupAffinity(GetCurrentProcess(), &GroupCount, GroupArrayAligned);

        if (status == 0 && GetLastError() != ERROR_INSUFFICIENT_BUFFER)
        {
            break;
        }

        if (status != 0)
        {
            return std::make_pair(status,
                                  std::vector(GroupArrayAligned, GroupArrayAligned + GroupCount));
        }
    }

    return std::make_pair(0, std::vector<USHORT>());
}

// On Windows there are two ways to set affinity, and therefore 2 ways to get it.
// These are not consistent, so we have to check both. In some cases it is actually
// not possible to determine affinity. For example when two different threads have
// affinity on different processor groups, set using SetThreadAffinityMask, we cannot
// retrieve the actual affinities.
// From documentation on GetProcessAffinityMask:
//     > If the calling process contains threads in multiple groups,
//     > the function returns zero for both affinity masks.
// In such cases we just give up and assume we have affinity for all processors.
// nullopt means no affinity is set, that is, all processors are allowed
WindowsAffinity get_process_affinity() {
    HMODULE k32                            = GetModuleHandle(TEXT("Kernel32.dll"));
    auto    GetThreadSelectedCpuSetMasks_f = GetThreadSelectedCpuSetMasks_t(
      (void (*)()) GetProcAddress(k32, "GetThreadSelectedCpuSetMasks"));

    BOOL status = 0;

    WindowsAffinity affinity;

    if (GetThreadSelectedCpuSetMasks_f != nullptr)
    {
        USHORT RequiredMaskCount;
        status = GetThreadSelectedCpuSetMasks_f(GetCurrentThread(), nullptr, 0, &RequiredMaskCount);

        // We expect ERROR_INSUFFICIENT_BUFFER from GetThreadSelectedCpuSetMasks,
        // but other failure is an actual error.
        if (status == 0 && GetLastError() != ERROR_INSUFFICIENT_BUFFER)
        {
            affinity.isNewDeterminate = false;
        }
        else if (RequiredMaskCount > 0)
        {
            // If RequiredMaskCount then these affinities were never set, but it's
            // not consistent so GetProcessAffinityMask may still return some affinity.
            auto groupAffinities = std::make_unique<GROUP_AFFINITY[]>(RequiredMaskCount);

            status = GetThreadSelectedCpuSetMasks_f(GetCurrentThread(), groupAffinities.get(),
                                                    RequiredMaskCount, &RequiredMaskCount);

            if (status == 0)
            {
                affinity.isNewDeterminate = false;
            }
            else
            {
                std::set<CpuIndex> cpus;

                for (USHORT i = 0; i < RequiredMaskCount; ++i)
                {
                    const size_t procGroupIndex = groupAffinities[i].Group;

                    for (size_t j = 0; j < WIN_PROCESSOR_GROUP_SIZE; ++j)
                    {
                        if (groupAffinities[i].Mask & (KAFFINITY(1) << j))
                            cpus.insert(procGroupIndex * WIN_PROCESSOR_GROUP_SIZE + j);
                    }
                }

                affinity.newApi = std::move(cpus);
            }
        }
    }

    // NOTE: There is no way to determine full affinity using the old API if
    //       individual threads set affinity on different processor groups.

    DWORD_PTR proc, sys;
    status = GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys);

    // If proc == 0 then we cannot determine affinity because it spans processor groups.
    // On Windows 11 and Server 2022 it will instead
    //     > If, however, hHandle specifies a handle to the current process, the function
    //     > always uses the calling thread's primary group (which by default is the same
    //     > as the process' primary group) in order to set the
    //     > lpProcessAffinityMask and lpSystemAffinityMask.
    // So it will never be indeterminate here. We can only make assumptions later.
    if (status == 0 || proc == 0)
    {
        affinity.isOldDeterminate = false;
        return affinity;
    }

    // If SetProcessAffinityMask was never called the affinity must span
    // all processor groups, but if it was called it must only span one.

    std::vector<USHORT> groupAffinity;  // We need to capture this later and capturing
                                        // from structured bindings requires c++20.

    std::tie(status, groupAffinity) = get_process_group_affinity();
    if (status == 0)
    {
        affinity.isOldDeterminate = false;
        return affinity;
    }

    if (groupAffinity.size() == 1)
    {
        // We detect the case when affinity is set to all processors and correctly
        // leave affinity.oldApi as nullopt.
        if (GetActiveProcessorGroupCount() != 1 || proc != sys)
        {
            std::set<CpuIndex> cpus;

            const size_t procGroupIndex = groupAffinity[0];

            const uint64_t mask = static_cast<uint64_t>(proc);
            for (size_t j = 0; j < WIN_PROCESSOR_GROUP_SIZE; ++j)
            {
                if (mask & (KAFFINITY(1) << j))
                    cpus.insert(procGroupIndex * WIN_PROCESSOR_GROUP_SIZE + j);
            }

            affinity.oldApi = std::move(cpus);
        }
    }
    else
    {
        // If we got here it means that either SetProcessAffinityMask was never set
        // or we're on Windows 11/Server 2022.

        // Since Windows 11 and Windows Server 2022 the behaviour of
        // GetProcessAffinityMask changed:
        //     > If, however, hHandle specifies a handle to the current process,
        //     > the function always uses the calling thread's primary group
        //     > (which by default is the same as the process' primary group)
        //     > in order to set the lpProcessAffinityMask and lpSystemAffinityMask.
        // In which case we can actually retrieve the full affinity.

        if (GetThreadSelectedCpuSetMasks_f != nullptr)
        {
            std::thread th([&]() {
                std::set<CpuIndex> cpus;
                bool               isAffinityFull = true;

                for (auto procGroupIndex : groupAffinity)
                {
                    const int numActiveProcessors =
                      GetActiveProcessorCount(static_cast<WORD>(procGroupIndex));

                    // We have to schedule to two different processors
                    // and & the affinities we get. Otherwise our processor
                    // choice could influence the resulting affinity.
                    // We assume the processor IDs within the group are
                    // filled sequentially from 0.
                    uint64_t procCombined = std::numeric_limits<uint64_t>::max();
                    uint64_t sysCombined  = std::numeric_limits<uint64_t>::max();

                    for (int i = 0; i < std::min(numActiveProcessors, 2); ++i)
                    {
                        GROUP_AFFINITY GroupAffinity;
                        std::memset(&GroupAffinity, 0, sizeof(GROUP_AFFINITY));
                        GroupAffinity.Group = static_cast<WORD>(procGroupIndex);

                        GroupAffinity.Mask = static_cast<KAFFINITY>(1) << i;

                        status =
                          SetThreadGroupAffinity(GetCurrentThread(), &GroupAffinity, nullptr);
                        if (status == 0)
                        {
                            affinity.isOldDeterminate = false;
                            return;
                        }

                        SwitchToThread();

                        DWORD_PTR proc2, sys2;
                        status = GetProcessAffinityMask(GetCurrentProcess(), &proc2, &sys2);
                        if (status == 0)
                        {
                            affinity.isOldDeterminate = false;
                            return;
                        }

                        procCombined &= static_cast<uint64_t>(proc2);
                        sysCombined &= static_cast<uint64_t>(sys2);
                    }

                    if (procCombined != sysCombined)
                        isAffinityFull = false;

                    for (size_t j = 0; j < WIN_PROCESSOR_GROUP_SIZE; ++j)
                    {
                        if (procCombined & (KAFFINITY(1) << j))
                            cpus.insert(procGroupIndex * WIN_PROCESSOR_GROUP_SIZE + j);
                    }
                }

                // We have to detect the case where the affinity was not set,
                // or is set to all processors so that we correctly produce as
                // std::nullopt result.
                if (!isAffinityFull)
                {
                    affinity.oldApi = std::move(cpus);
                }
            });

            th.join();
        }
    }

    return affinity;
}

// Type machinery used to emulate Cache->GroupCount

template<typename T, typename = void>
struct HasGroupCount: std::false_type {};

template<typename T>
struct HasGroupCount<T, std::void_t<decltype(std::declval<T>().Cache.GroupCount)>>: std::true_type {
};

template<typename T, typename Pred, std::enable_if_t<HasGroupCount<T>::value, bool> = true>
std::set<CpuIndex> readCacheMembers(const T* info, Pred&& is_cpu_allowed) {
    std::set<CpuIndex> cpus;
    // On Windows 10 this will read a 0 because GroupCount doesn't exist
    int groupCount = std::max(info->Cache.GroupCount, WORD(1));
    for (WORD procGroup = 0; procGroup < groupCount; ++procGroup)
    {
        for (BYTE number = 0; number < WIN_PROCESSOR_GROUP_SIZE; ++number)
        {
            WORD           groupNumber = info->Cache.GroupMasks[procGroup].Group;
            const CpuIndex c = static_cast<CpuIndex>(groupNumber) * WIN_PROCESSOR_GROUP_SIZE
                             + static_cast<CpuIndex>(number);
            if (!(info->Cache.GroupMasks[procGroup].Mask & (1ULL << number)) || !is_cpu_allowed(c))
                continue;
            cpus.insert(c);
        }
    }
    return cpus;
}

template<typename T, typename Pred, std::enable_if_t<!HasGroupCount<T>::value, bool> = true>
std::set<CpuIndex> readCacheMembers(const T* info, Pred&& is_cpu_allowed) {
    std::set<CpuIndex> cpus;
    for (BYTE number = 0; number < WIN_PROCESSOR_GROUP_SIZE; ++number)
    {
        WORD           groupNumber = info->Cache.GroupMask.Group;
        const CpuIndex c           = static_cast<CpuIndex>(groupNumber) * WIN_PROCESSOR_GROUP_SIZE
                         + static_cast<CpuIndex>(number);
        if (!(info->Cache.GroupMask.Mask & (1ULL << number)) || !is_cpu_allowed(c))
            continue;
        cpus.insert(c);
    }
    return cpus;
}

#endif

#if defined(__linux__) && !defined(__ANDROID__)

std::set<CpuIndex> get_process_affinity() {

    std::set<CpuIndex> cpus;

    // For unsupported systems, or in case of a soft error, we may assume
    // all processors are available for use.
    [[maybe_unused]] auto set_to_all_cpus = [&]() {
        for (CpuIndex c = 0; c < SYSTEM_THREADS_NB; ++c)
            cpus.insert(c);
    };

    // cpu_set_t by default holds 1024 entries. This may not be enough soon,
    // but there is no easy way to determine how many threads there actually
    // is. In this case we just choose a reasonable upper bound.
    static constexpr CpuIndex MaxNumCpus = 1024 * 64;

    cpu_set_t* mask = CPU_ALLOC(MaxNumCpus);
    if (mask == nullptr)
        std::exit(EXIT_FAILURE);

    const size_t masksize = CPU_ALLOC_SIZE(MaxNumCpus);

    CPU_ZERO_S(masksize, mask);

    const int status = sched_getaffinity(0, masksize, mask);

    if (status != 0)
    {
        CPU_FREE(mask);
        std::exit(EXIT_FAILURE);
    }

    for (CpuIndex c = 0; c < MaxNumCpus; ++c)
        if (CPU_ISSET_S(c, masksize, mask))
            cpus.insert(c);

    CPU_FREE(mask);

    return cpus;
}

#endif

#if defined(__linux__) && !defined(__ANDROID__)

const std::set<CpuIndex> STARTUP_PROCESSOR_AFFINITY = get_process_affinity();

#elif defined(_WIN64)

const WindowsAffinity STARTUP_PROCESSOR_AFFINITY = get_process_affinity();
const bool STARTUP_USE_OLD_AFFINITY_API =
  STARTUP_PROCESSOR_AFFINITY.likely_used_old_api();

#endif

NumaConfig::NumaConfig() :
    highestCpuIndex(0),
    customAffinity(false) {
    const auto numCpus = SYSTEM_THREADS_NB;
    add_cpu_range_to_node(NumaIndex{0}, CpuIndex{0}, numCpus - 1);
}

// This function gets a NumaConfig based on the system's provided information.
// The available policies are documented above.
NumaConfig NumaConfig::from_system([[maybe_unused]] const NumaAutoPolicy& policy,
                                  bool respectProcessAffinity) {
    NumaConfig cfg = empty();

#if !((defined(__linux__) && !defined(__ANDROID__)) || defined(_WIN64))
    // Fallback for unsupported systems.
    for (CpuIndex c = 0; c < SYSTEM_THREADS_NB; ++c)
        cfg.add_cpu_to_node(NumaIndex{0}, c);
#else

    #if defined(_WIN64)

    std::optional<std::set<CpuIndex>> allowedCpus;

    if (respectProcessAffinity)
        allowedCpus = STARTUP_PROCESSOR_AFFINITY.get_combined();

    // The affinity cannot be determined in all cases on Windows,
    // but we at least guarantee that the number of allowed processors
    // is >= number of processors in the affinity mask. In case the user
    // is not satisfied they must set the processor numbers explicitly.
    auto is_cpu_allowed = [&allowedCpus](CpuIndex c) {
        return !allowedCpus.has_value() || allowedCpus->count(c) == 1;
    };

    #elif defined(__linux__) && !defined(__ANDROID__)

    std::set<CpuIndex> allowedCpus;

    if (respectProcessAffinity)
        allowedCpus = STARTUP_PROCESSOR_AFFINITY;

    auto is_cpu_allowed = [respectProcessAffinity, &allowedCpus](CpuIndex c) {
        return !respectProcessAffinity || allowedCpus.count(c) == 1;
    };

    #endif

    bool l3Success = false;
    if (!std::holds_alternative<SystemNumaPolicy>(policy))
    {
        size_t l3BundleSize = 0;
        if (const auto* v = std::get_if<BundledL3Policy>(&policy))
        {
            l3BundleSize = v->bundleSize;
        }
        if (auto l3Cfg =
              try_get_l3_aware_config(respectProcessAffinity, l3BundleSize, is_cpu_allowed))
        {
            cfg       = std::move(*l3Cfg);
            l3Success = true;
        }
    }
    if (!l3Success)
        cfg = from_system_numa(respectProcessAffinity, is_cpu_allowed);

    #if defined(_WIN64)
    // Split the NUMA nodes to be contained within a group if necessary.
    // This is needed between Windows 10 Build 20348 and Windows 11, because
    // the new NUMA allocation behaviour was introduced while there was
    // still no way to set thread affinity spanning multiple processor groups.
    // See https://learn.microsoft.com/en-us/windows/win32/procthread/numa-support
    // We also do this is if need to force old API for some reason.
    //
    // 2024-08-26: It appears that we need to actually always force this behaviour.
    // While Windows allows this to work now, such assignments have bad interaction
    // with the scheduler - in particular it still prefers scheduling on the thread's
    // "primary" node, even if it means scheduling SMT processors first.
    // See https://github.com/official-stockfish/Stockfish/issues/5551
    // See https://learn.microsoft.com/en-us/windows/win32/procthread/processor-groups
    //
    //     Each process is assigned a primary group at creation, and by default all
    //     of its threads' primary group is the same. Each thread's ideal processor
    //     is in the thread's primary group, so threads will preferentially be
    //     scheduled to processors on their primary group, but they are able to
    //     be scheduled to processors on any other group.
    //
    // used to be guarded by if (STARTUP_USE_OLD_AFFINITY_API)
    {
        NumaConfig splitCfg = empty();

        NumaIndex splitNodeIndex = 0;
        for (const auto& cpus : cfg.nodes)
        {
            if (cpus.empty())
                continue;

            size_t lastProcGroupIndex = *(cpus.begin()) / WIN_PROCESSOR_GROUP_SIZE;
            for (CpuIndex c : cpus)
            {
                const size_t procGroupIndex = c / WIN_PROCESSOR_GROUP_SIZE;
                if (procGroupIndex != lastProcGroupIndex)
                {
                    splitNodeIndex += 1;
                    lastProcGroupIndex = procGroupIndex;
                }
                splitCfg.add_cpu_to_node(splitNodeIndex, c);
            }
            splitNodeIndex += 1;
        }

        cfg = std::move(splitCfg);
    }
    #endif

#endif

    // We have to ensure no empty NUMA nodes persist.
    cfg.remove_empty_numa_nodes();

    // If the user explicitly opts out from respecting the current process affinity
    // then it may be inconsistent with the current affinity (obviously), so we
    // consider it custom.
    if (!respectProcessAffinity)
        cfg.customAffinity = true;

    return cfg;
}

// ':'-separated numa nodes
// ','-separated cpu indices
// supports "first-last" range syntax for cpu indices
// For example "0-15,128-143:16-31,144-159:32-47,160-175:48-63,176-191"
NumaConfig NumaConfig::from_string(const std::string& s) {
    NumaConfig cfg = empty();

    NumaIndex n = 0;
    for (auto&& nodeStr : split(s, ":"))
    {
        auto indices = indices_from_shortened_string(std::string(nodeStr));
        if (!indices.empty())
        {
            for (auto idx : indices)
            {
                if (!cfg.add_cpu_to_node(n, CpuIndex(idx)))
                    std::exit(EXIT_FAILURE);
            }

            n += 1;
        }
    }

    cfg.customAffinity = true;

    return cfg;
}

std::string NumaConfig::to_string() const {
    std::string str;

    bool isFirstNode = true;
    for (auto&& cpus : nodes)
    {
        if (!isFirstNode)
            str += ":";

        bool isFirstSet = true;
        auto rangeStart = cpus.begin();
        for (auto it = cpus.begin(); it != cpus.end(); ++it)
        {
            auto next = std::next(it);
            if (next == cpus.end() || *next != *it + 1)
            {
                // cpus[i] is at the end of the range (may be of size 1)
                if (!isFirstSet)
                    str += ",";

                const CpuIndex last = *it;

                if (it != rangeStart)
                {
                    const CpuIndex first = *rangeStart;

                    str += std::to_string(first);
                    str += "-";
                    str += std::to_string(last);
                }
                else
                    str += std::to_string(last);

                rangeStart = next;
                isFirstSet = false;
            }
        }

        isFirstNode = false;
    }

    return str;
}

bool NumaConfig::suggests_binding_threads(CpuIndex numThreads) const {
    // If we can reasonably determine that the threads cannot be contained
    // by the OS within the first NUMA node then we advise distributing
    // and binding threads. When the threads are not bound we can only use
    // NUMA memory replicated objects from the first node, so when the OS
    // has to schedule on other nodes we lose performance. We also suggest
    // binding if there's enough threads to distribute among nodes with minimal
    // disparity. We try to ignore small nodes, in particular the empty ones.

    // If the affinity set by the user does not match the affinity given by
    // the OS then binding is necessary to ensure the threads are running on
    // correct processors.
    if (customAffinity)
        return true;

    // We obviously cannot distribute a single thread, so a single thread
    // should never be bound.
    if (numThreads <= 1)
        return false;

    size_t largestNodeSize = 0;
    for (auto&& cpus : nodes)
        if (cpus.size() > largestNodeSize)
            largestNodeSize = cpus.size();

    auto is_node_small = [largestNodeSize](const std::set<CpuIndex>& node) {
        static constexpr double SmallNodeThreshold = 0.6;
        return static_cast<double>(node.size()) / static_cast<double>(largestNodeSize)
            <= SmallNodeThreshold;
    };

    size_t numNotSmallNodes = 0;
    for (auto&& cpus : nodes)
        if (!is_node_small(cpus))
            numNotSmallNodes += 1;

    return (numThreads > largestNodeSize / 2 || numThreads >= numNotSmallNodes * 4)
        && nodes.size() > 1;
}

std::vector<NumaIndex> NumaConfig::distribute_threads_among_numa_nodes(CpuIndex numThreads) const {
    std::vector<NumaIndex> ns;

    if (nodes.size() == 1)
    {
        // Special case for when there's no NUMA nodes. This doesn't buy us
        // much, but let's keep the default path simple.
        ns.resize(numThreads, NumaIndex{0});
    }
    else
    {
        std::vector<size_t> occupation(nodes.size(), 0);
        for (CpuIndex c = 0; c < numThreads; ++c)
        {
            NumaIndex bestNode{0};
            float     bestNodeFill = std::numeric_limits<float>::max();
            for (NumaIndex n = 0; n < nodes.size(); ++n)
            {
                float fill =
                  static_cast<float>(occupation[n] + 1) / static_cast<float>(nodes[n].size());
                // NOTE: Do we want to perhaps fill the first available node
                //       up to 50% first before considering other nodes?
                //       Probably not, because it would interfere with running
                //       multiple instances. We basically shouldn't favor any
                //       particular node.
                if (fill < bestNodeFill)
                {
                    bestNode     = n;
                    bestNodeFill = fill;
                }
            }
            ns.emplace_back(bestNode);
            occupation[bestNode] += 1;
        }
    }

    return ns;
}

NumaReplicatedAccessToken NumaConfig::bind_current_thread_to_numa_node(NumaIndex n) const {
    if (n >= nodes.size() || nodes[n].size() == 0)
        std::exit(EXIT_FAILURE);

#if defined(__linux__) && !defined(__ANDROID__)

    cpu_set_t* mask = CPU_ALLOC(highestCpuIndex + 1);
    if (mask == nullptr)
        std::exit(EXIT_FAILURE);

    const size_t masksize = CPU_ALLOC_SIZE(highestCpuIndex + 1);

    CPU_ZERO_S(masksize, mask);

    for (CpuIndex c : nodes[n])
        CPU_SET_S(c, masksize, mask);

    const int status = sched_setaffinity(0, masksize, mask);

    CPU_FREE(mask);

    if (status != 0)
        std::exit(EXIT_FAILURE);

    // We yield this thread just to be sure it gets rescheduled.
    // This is defensive, allowed because this code is not performance critical.
    sched_yield();

#elif defined(_WIN64)

    // Requires Windows 11. No good way to set thread affinity spanning
    // processor groups before that.
    HMODULE k32                            = GetModuleHandle(TEXT("Kernel32.dll"));
    auto    SetThreadSelectedCpuSetMasks_f = SetThreadSelectedCpuSetMasks_t(
      (void (*)()) GetProcAddress(k32, "GetThreadSelectedCpuSetMasks"));

    // We ALWAYS set affinity with the new API if available, because
    // there's no downsides, and we forcibly keep it consistent with
    // the old API should we need to use it. I.e. we always keep this
    // as a superset of what we set with SetThreadGroupAffinity.
    if (SetThreadSelectedCpuSetMasks_f != nullptr)
    {
        // Only available on Windows 11 and Windows Server 2022 onwards
        const USHORT numProcGroups = USHORT(
          ((highestCpuIndex + 1) + WIN_PROCESSOR_GROUP_SIZE - 1) / WIN_PROCESSOR_GROUP_SIZE);
        auto groupAffinities = std::make_unique<GROUP_AFFINITY[]>(numProcGroups);
        std::memset(groupAffinities.get(), 0, sizeof(GROUP_AFFINITY) * numProcGroups);
        for (WORD i = 0; i < numProcGroups; ++i)
            groupAffinities[i].Group = i;

        for (CpuIndex c : nodes[n])
        {
            const size_t procGroupIndex     = c / WIN_PROCESSOR_GROUP_SIZE;
            const size_t idxWithinProcGroup = c % WIN_PROCESSOR_GROUP_SIZE;
            groupAffinities[procGroupIndex].Mask |= KAFFINITY(1) << idxWithinProcGroup;
        }

        HANDLE hThread = GetCurrentThread();

        const BOOL status =
          SetThreadSelectedCpuSetMasks_f(hThread, groupAffinities.get(), numProcGroups);
        if (status == 0)
            std::exit(EXIT_FAILURE);

        // We yield this thread just to be sure it gets rescheduled.
        // This is defensive, allowed because this code is not performance critical.
        SwitchToThread();
    }

    // Sometimes we need to force the old API, but do not use it unless necessary.
    if (SetThreadSelectedCpuSetMasks_f == nullptr || STARTUP_USE_OLD_AFFINITY_API)
    {
        // On earlier windows version (since windows 7) we cannot run a single thread
        // on multiple processor groups, so we need to restrict the group.
        // We assume the group of the first processor listed for this node.
        // Processors from outside this group will not be assigned for this thread.
        // Normally this won't be an issue because windows used to assign NUMA nodes
        // such that they cannot span processor groups. However, since Windows 10
        // Build 20348 the behaviour changed, so there's a small window of versions
        // between this and Windows 11 that might exhibit problems with not all
        // processors being utilized.
        //
        // We handle this in NumaConfig::from_system by manually splitting the
        // nodes when we detect that there is no function to set affinity spanning
        // processor nodes. This is required because otherwise our thread distribution
        // code may produce suboptimal results.
        //
        // See https://learn.microsoft.com/en-us/windows/win32/procthread/numa-support
        GROUP_AFFINITY affinity;
        std::memset(&affinity, 0, sizeof(GROUP_AFFINITY));
        // We use an ordered set to be sure to get the smallest cpu number here.
        const size_t forcedProcGroupIndex = *(nodes[n].begin()) / WIN_PROCESSOR_GROUP_SIZE;
        affinity.Group                    = static_cast<WORD>(forcedProcGroupIndex);
        for (CpuIndex c : nodes[n])
        {
            const size_t procGroupIndex     = c / WIN_PROCESSOR_GROUP_SIZE;
            const size_t idxWithinProcGroup = c % WIN_PROCESSOR_GROUP_SIZE;
            // We skip processors that are not in the same processor group.
            // If everything was set up correctly this will never be an issue,
            // but we have to account for bad NUMA node specification.
            if (procGroupIndex != forcedProcGroupIndex)
                continue;

            affinity.Mask |= KAFFINITY(1) << idxWithinProcGroup;
        }

        HANDLE hThread = GetCurrentThread();

        const BOOL status = SetThreadGroupAffinity(hThread, &affinity, nullptr);
        if (status == 0)
            std::exit(EXIT_FAILURE);

        // We yield this thread just to be sure it gets rescheduled. This is
        // defensive, allowed because this code is not performance critical.
        SwitchToThread();
    }

#endif

    return NumaReplicatedAccessToken(n);
}

void NumaConfig::remove_empty_numa_nodes() {
    std::vector<std::set<CpuIndex>> newNodes;
    for (auto&& cpus : nodes)
        if (!cpus.empty())
            newNodes.emplace_back(std::move(cpus));
    nodes = std::move(newNodes);
}

// Returns true if successful
// Returns false if failed, i.e. when the cpu is already present
//                          strong guarantee, the structure remains unmodified
bool NumaConfig::add_cpu_to_node(NumaIndex n, CpuIndex c) {
    if (is_cpu_assigned(c))
        return false;

    while (nodes.size() <= n)
        nodes.emplace_back();

    nodes[n].insert(c);
    nodeByCpu[c] = n;

    if (c > highestCpuIndex)
        highestCpuIndex = c;

    return true;
}

// Returns true if successful
// Returns false if failed, i.e. when any of the cpus is already present
//                          strong guarantee, the structure remains unmodified
bool NumaConfig::add_cpu_range_to_node(NumaIndex n, CpuIndex cfirst, CpuIndex clast) {
    for (CpuIndex c = cfirst; c <= clast; ++c)
        if (is_cpu_assigned(c))
            return false;

    while (nodes.size() <= n)
        nodes.emplace_back();

    for (CpuIndex c = cfirst; c <= clast; ++c)
    {
        nodes[n].insert(c);
        nodeByCpu[c] = n;
    }

    if (clast > highestCpuIndex)
        highestCpuIndex = clast;

    return true;
}

std::vector<size_t> NumaConfig::indices_from_shortened_string(const std::string& s) {
    std::vector<size_t> indices;

    if (s.empty())
        return indices;

    for (const auto& ss : split(s, ","))
    {
        if (ss.empty())
            continue;

        auto parts = split(ss, "-");
        if (parts.size() == 1)
        {
            const CpuIndex c = CpuIndex{str_to_size_t(std::string(parts[0]))};
            indices.emplace_back(c);
        }
        else if (parts.size() == 2)
        {
            const CpuIndex cfirst = CpuIndex{str_to_size_t(std::string(parts[0]))};
            const CpuIndex clast  = CpuIndex{str_to_size_t(std::string(parts[1]))};
            for (size_t c = cfirst; c <= clast; ++c)
            {
                indices.emplace_back(c);
            }
        }
    }

    return indices;
}

// This function queries the system for the mapping of processors to NUMA nodes.
// On Linux we read from standardized kernel sysfs, with a fallback to single NUMA
// node. On Windows we utilize GetNumaProcessorNodeEx, which has its quirks, see
// comment for Windows implementation of get_process_affinity.
NumaConfig NumaConfig::from_system_numa([[maybe_unused]] bool respectProcessAffinity,
                                   std::function<bool(CpuIndex)> is_cpu_allowed) {
    NumaConfig cfg = empty();

#if defined(__linux__) && !defined(__ANDROID__)

    // On Linux things are straightforward, since there's no processor groups and
    // any thread can be scheduled on all processors.
    // We try to gather this information from the sysfs first
    // https://www.kernel.org/doc/Documentation/ABI/stable/sysfs-devices-node

    bool useFallback = false;
    auto fallback    = [&]() {
        useFallback = true;
        cfg         = empty();
    };

    // /sys/devices/system/node/online contains information about active NUMA nodes
    auto nodeIdsStr = read_file_to_string("/sys/devices/system/node/online");
    if (!nodeIdsStr.has_value() || nodeIdsStr->empty())
    {
        fallback();
    }
    else
    {
        remove_whitespace(*nodeIdsStr);
        for (size_t n : indices_from_shortened_string(*nodeIdsStr))
        {
            // /sys/devices/system/node/node.../cpulist
            std::string path =
              std::string("/sys/devices/system/node/node") + std::to_string(n) + "/cpulist";
            auto cpuIdsStr = read_file_to_string(path);
            // Now, we only bail if the file does not exist. Some nodes may be
            // empty, that's fine. An empty node still has a file that appears
            // to have some whitespace, so we need to handle that.
            if (!cpuIdsStr.has_value())
            {
                fallback();
                break;
            }
            else
            {
                remove_whitespace(*cpuIdsStr);
                for (size_t c : indices_from_shortened_string(*cpuIdsStr))
                {
                    if (is_cpu_allowed(c))
                        cfg.add_cpu_to_node(n, c);
                }
            }
        }
    }

    if (useFallback)
    {
        for (CpuIndex c = 0; c < SYSTEM_THREADS_NB; ++c)
            if (is_cpu_allowed(c))
                cfg.add_cpu_to_node(NumaIndex{0}, c);
    }

#elif defined(_WIN64)

    WORD numProcGroups = GetActiveProcessorGroupCount();
    for (WORD procGroup = 0; procGroup < numProcGroups; ++procGroup)
    {
        for (BYTE number = 0; number < WIN_PROCESSOR_GROUP_SIZE; ++number)
        {
            PROCESSOR_NUMBER procnum;
            procnum.Group    = procGroup;
            procnum.Number   = number;
            procnum.Reserved = 0;
            USHORT nodeNumber;

            const BOOL     status = GetNumaProcessorNodeEx(&procnum, &nodeNumber);
            const CpuIndex c      = static_cast<CpuIndex>(procGroup) * WIN_PROCESSOR_GROUP_SIZE
                                 + static_cast<CpuIndex>(number);
            if (status != 0 && nodeNumber != std::numeric_limits<USHORT>::max()
                && is_cpu_allowed(c))
            {
                cfg.add_cpu_to_node(nodeNumber, c);
            }
        }
    }

#endif

    return cfg;
}

std::optional<NumaConfig> NumaConfig::try_get_l3_aware_config(
  bool respectProcessAffinity, size_t bundleSize, std::function<bool(CpuIndex)> is_cpu_allowed) {
    // Get the normal system configuration so we know to which NUMA node
    // each L3 domain belongs.
    NumaConfig systemConfig =
      NumaConfig::from_system(SystemNumaPolicy{}, respectProcessAffinity);
    std::vector<L3Domain> l3Domains;

#if defined(__linux__) && !defined(__ANDROID__)

    std::set<CpuIndex> seenCpus;
    auto               nextUnseenCpu = [&seenCpus]() {
        for (CpuIndex i = 0;; ++i)
            if (!seenCpus.count(i))
                return i;
    };

    while (true)
    {
        CpuIndex next = nextUnseenCpu();
        auto     siblingsStr =
          read_file_to_string("/sys/devices/system/cpu/cpu" + std::to_string(next)
                              + "/cache/index3/shared_cpu_list");

        if (!siblingsStr.has_value() || siblingsStr->empty())
        {
            break;  // we have read all available CPUs
        }

        L3Domain domain;
        for (size_t c : indices_from_shortened_string(*siblingsStr))
        {
            if (is_cpu_allowed(c))
            {
                domain.systemNumaIndex = systemConfig.nodeByCpu.at(c);
                domain.cpus.insert(c);
            }
            seenCpus.insert(c);
        }
        if (!domain.cpus.empty())
        {
            l3Domains.emplace_back(std::move(domain));
        }
    }

#elif defined(_WIN64)

    DWORD bufSize = 0;
    GetLogicalProcessorInformationEx(RelationCache, nullptr, &bufSize);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER)
        return std::nullopt;

    std::vector<char> buffer(bufSize);
    auto info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
    if (!GetLogicalProcessorInformationEx(RelationCache, info, &bufSize))
        return std::nullopt;

    while (reinterpret_cast<char*>(info) < buffer.data() + bufSize)
    {
        info = std::launder(info);
        if (info->Relationship == RelationCache && info->Cache.Level == 3)
        {
            L3Domain domain{};
            domain.cpus = readCacheMembers(info, is_cpu_allowed);
            if (!domain.cpus.empty())
            {
                domain.systemNumaIndex = systemConfig.nodeByCpu.at(*domain.cpus.begin());
                l3Domains.push_back(std::move(domain));
            }
        }
        // Variable length data structure, advance to next
        info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
          reinterpret_cast<char*>(info) + info->Size);
    }
#endif

    if (!l3Domains.empty())
        return {NumaConfig::from_l3_info(std::move(l3Domains), bundleSize)};

    return std::nullopt;
}

NumaConfig NumaConfig::from_l3_info(std::vector<L3Domain>&& domains, size_t bundleSize) {
    assert(!domains.empty());

    std::map<NumaIndex, std::vector<L3Domain>> list;
    for (auto& d : domains)
        list[d.systemNumaIndex].emplace_back(std::move(d));

    NumaConfig cfg = empty();
    NumaIndex  n   = 0;
    for (auto& [_, ds] : list)
    {
        bool changed;
        // Scan through pairs and merge them. With roughly equal L3 sizes, should give
        // a decent distribution.
        do
        {
            changed = false;
            for (size_t j = 0; j + 1 < ds.size(); ++j)
            {
                if (ds[j].cpus.size() + ds[j + 1].cpus.size() <= bundleSize)
                {
                    changed = true;
                    ds[j].cpus.merge(ds[j + 1].cpus);
                    ds.erase(ds.begin() + j + 1);
                }
            }
            // ds.size() has decreased if changed is true, so this loop will terminate
        } while (changed);
        for (const L3Domain& d : ds)
        {
            const NumaIndex dn = n++;
            for (CpuIndex cpu : d.cpus)
            {
                cfg.add_cpu_to_node(dn, cpu);
            }
        }
    }
    return cfg;
}

NumaReplicatedBase::NumaReplicatedBase(NumaReplicationContext& ctx) :
    context(&ctx) {
    context->attach(this);
}

NumaReplicatedBase::NumaReplicatedBase(NumaReplicatedBase&& other) noexcept :
    context(std::exchange(other.context, nullptr)) {
    context->move_attached(&other, this);
}

NumaReplicatedBase& NumaReplicatedBase::operator=(NumaReplicatedBase&& other) noexcept {
    context = std::exchange(other.context, nullptr);

    context->move_attached(&other, this);

    return *this;
}

NumaReplicatedBase::~NumaReplicatedBase() {
    if (context != nullptr)
        context->detach(this);
}

const NumaConfig& NumaReplicatedBase::get_numa_config() const {
    return context->get_numa_config();
}

}  // namespace Stockfish
