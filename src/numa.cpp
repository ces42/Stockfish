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
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__) && !defined(__ANDROID__)
    #include <sched.h>
#endif

namespace Stockfish {

CpuIndex get_hardware_concurrency() {
    CpuIndex concurrency = std::thread::hardware_concurrency();

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
    static constexpr size_t GroupArrayMinimumAlignment = 4;
    static_assert(GroupArrayMinimumAlignment >= alignof(USHORT));

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

        if (status == 0 && GetLastError() != ERROR_INSUFFICIENT_BUFFER)
        {
            affinity.isNewDeterminate = false;
        }
        else if (RequiredMaskCount > 0)
        {
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

    DWORD_PTR proc, sys;
    status = GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys);

    if (status == 0 || proc == 0)
    {
        affinity.isOldDeterminate = false;
        return affinity;
    }

    std::vector<USHORT> groupAffinity;

    std::tie(status, groupAffinity) = get_process_group_affinity();
    if (status == 0)
    {
        affinity.isOldDeterminate = false;
        return affinity;
    }

    if (groupAffinity.size() == 1)
    {
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
        if (GetThreadSelectedCpuSetMasks_f != nullptr)
        {
            std::thread th([&]() {
                std::set<CpuIndex> cpus;
                bool               isAffinityFull = true;

                for (auto procGroupIndex : groupAffinity)
                {
                    const int numActiveProcessors =
                      GetActiveProcessorCount(static_cast<WORD>(procGroupIndex));

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

std::set<CpuIndex> readCacheMembers(const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* info, std::function<bool(CpuIndex)> is_cpu_allowed) {
    std::set<CpuIndex> cpus;

    // Use a runtime check for GroupCount as it's not present in all Windows versions' headers.
    // However, for Stockfish we can assume a modern enough SDK.
    // The original code used a template trick to handle missing GroupCount.
    // Here we'll use the GroupCount if available in our build environment.
    int groupCount = std::max<int>(info->Cache.GroupCount, 1);
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

#endif

#if defined(__linux__) && !defined(__ANDROID__)

std::set<CpuIndex> get_process_affinity() {
    std::set<CpuIndex> cpus;

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
const bool STARTUP_USE_OLD_AFFINITY_API = STARTUP_PROCESSOR_AFFINITY.likely_used_old_api();
#endif

NumaConfig::NumaConfig() :
    highestCpuIndex(0),
    customAffinity(false) {
    const auto numCpus = SYSTEM_THREADS_NB;
    add_cpu_range_to_node(NumaIndex{0}, CpuIndex{0}, numCpus - 1);
}

NumaConfig NumaConfig::from_system(const NumaAutoPolicy& policy, bool respectProcessAffinity) {
    NumaConfig cfg = empty();

#if !((defined(__linux__) && !defined(__ANDROID__)) || defined(_WIN64))
    for (CpuIndex c = 0; c < SYSTEM_THREADS_NB; ++c)
        cfg.add_cpu_to_node(NumaIndex{0}, c);
#else

    #if defined(_WIN64)
    std::optional<std::set<CpuIndex>> allowedCpus;
    if (respectProcessAffinity)
        allowedCpus = STARTUP_PROCESSOR_AFFINITY.get_combined();
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

    cfg.remove_empty_numa_nodes();

    if (!respectProcessAffinity)
        cfg.customAffinity = true;

    return cfg;
}

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
    if (customAffinity)
        return true;

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

    sched_yield();

#elif defined(_WIN64)

    HMODULE k32                            = GetModuleHandle(TEXT("Kernel32.dll"));
    auto    SetThreadSelectedCpuSetMasks_f = SetThreadSelectedCpuSetMasks_t(
      (void (*)()) GetProcAddress(k32, "GetThreadSelectedCpuSetMasks"));

    if (SetThreadSelectedCpuSetMasks_f != nullptr)
    {
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

        SwitchToThread();
    }

    if (SetThreadSelectedCpuSetMasks_f == nullptr || STARTUP_USE_OLD_AFFINITY_API)
    {
        GROUP_AFFINITY affinity;
        std::memset(&affinity, 0, sizeof(GROUP_AFFINITY));
        const size_t forcedProcGroupIndex = *(nodes[n].begin()) / WIN_PROCESSOR_GROUP_SIZE;
        affinity.Group                    = static_cast<WORD>(forcedProcGroupIndex);
        for (CpuIndex c : nodes[n])
        {
            const size_t procGroupIndex     = c / WIN_PROCESSOR_GROUP_SIZE;
            const size_t idxWithinProcGroup = c % WIN_PROCESSOR_GROUP_SIZE;
            if (procGroupIndex != forcedProcGroupIndex)
                continue;

            affinity.Mask |= KAFFINITY(1) << idxWithinProcGroup;
        }

        HANDLE hThread = GetCurrentThread();

        const BOOL status = SetThreadGroupAffinity(hThread, &affinity, nullptr);
        if (status == 0)
            std::exit(EXIT_FAILURE);

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

NumaConfig NumaConfig::from_system_numa([[maybe_unused]] bool respectProcessAffinity, std::function<bool(CpuIndex)> is_cpu_allowed) {
    NumaConfig cfg = empty();

#if defined(__linux__) && !defined(__ANDROID__)
    bool useFallback = false;
    auto fallback    = [&]() {
        useFallback = true;
        cfg         = empty();
    };

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
            std::string path =
              std::string("/sys/devices/system/node/node") + std::to_string(n) + "/cpulist";
            auto cpuIdsStr = read_file_to_string(path);
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
            break;
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
