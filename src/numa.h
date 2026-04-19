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

#ifndef NUMA_H_INCLUDED
#define NUMA_H_INCLUDED

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>
#include <cstring>
#include <optional>

#include "shm.h"

// We support linux very well, but we explicitly do NOT support Android,
// because there is no affected systems, not worth maintaining.
#if defined(__linux__) && !defined(__ANDROID__)
    #if !defined(_GNU_SOURCE)
        #define _GNU_SOURCE
    #endif
    #include <sched.h>
#elif defined(_WIN64)

    #if _WIN32_WINNT < 0x0601
        #undef _WIN32_WINNT
        #define _WIN32_WINNT 0x0601  // Force to include needed API prototypes
    #endif

// On Windows each processor group can have up to 64 processors.
// https://learn.microsoft.com/en-us/windows/win32/procthread/processor-groups
static constexpr size_t WIN_PROCESSOR_GROUP_SIZE = 64;

    #if !defined(NOMINMAX)
        #define NOMINMAX
    #endif
    #include <windows.h>
    #if defined small
        #undef small
    #endif

// https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setthreadselectedcpusetmasks
using SetThreadSelectedCpuSetMasks_t = BOOL (*)(HANDLE, PGROUP_AFFINITY, USHORT);

// https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getthreadselectedcpusetmasks
using GetThreadSelectedCpuSetMasks_t = BOOL (*)(HANDLE, PGROUP_AFFINITY, USHORT, PUSHORT);

#endif

#include "misc.h"

namespace Stockfish {

using CpuIndex  = size_t;
using NumaIndex = size_t;

CpuIndex get_hardware_concurrency();

extern const CpuIndex SYSTEM_THREADS_NB;

#if defined(_WIN64)

struct WindowsAffinity {
    std::optional<std::set<CpuIndex>> oldApi;
    std::optional<std::set<CpuIndex>> newApi;

    // We also provide diagnostic for when the affinity is set to nullopt
    // whether it was due to being indeterminate. If affinity is indeterminate
    // it is best to assume it is not set at all, so consistent with the meaning
    // of the nullopt affinity.
    bool isNewDeterminate = true;
    bool isOldDeterminate = true;

    std::optional<std::set<CpuIndex>> get_combined() const;

    // Since Windows 11 and Windows Server 2022 thread affinities can span
    // processor groups and can be set as such by a new WinAPI function. However,
    // we may need to force using the old API if we detect that the process has
    // affinity set by the old API already and we want to override that. Due to the
    // limitations of the old API we cannot detect its use reliably. There will be
    // cases where we detect not use but it has actually been used and vice versa.

    bool likely_used_old_api() const { return oldApi.has_value() || !isOldDeterminate; }
};

std::pair<BOOL, std::vector<USHORT>> get_process_group_affinity();

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
WindowsAffinity get_process_affinity();

#endif

#if defined(__linux__) && !defined(__ANDROID__)

std::set<CpuIndex> get_process_affinity();

#endif

#if defined(__linux__) && !defined(__ANDROID__)

extern const std::set<CpuIndex> STARTUP_PROCESSOR_AFFINITY;

#elif defined(_WIN64)

extern const WindowsAffinity STARTUP_PROCESSOR_AFFINITY;
extern const bool STARTUP_USE_OLD_AFFINITY_API;

#endif

// We want to abstract the purpose of storing the numa node index somewhat.
// Whoever is using this does not need to know the specifics of the replication
// machinery to be able to access NUMA replicated memory.
class NumaReplicatedAccessToken {
   public:
    NumaReplicatedAccessToken() :
        n(0) {}

    explicit NumaReplicatedAccessToken(NumaIndex idx) :
        n(idx) {}

    NumaIndex get_numa_index() const { return n; }

   private:
    NumaIndex n;
};

struct L3Domain {
    NumaIndex          systemNumaIndex{};
    std::set<CpuIndex> cpus{};
};

// Use system NUMA nodes
struct SystemNumaPolicy {};
// Use system-reported L3 domains
struct L3DomainsPolicy {};
// Group system-reported L3 domains until they reach bundleSize
struct BundledL3Policy {
    size_t bundleSize;
};

using NumaAutoPolicy = std::variant<SystemNumaPolicy, L3DomainsPolicy, BundledL3Policy>;

// Designed as immutable, because there is no good reason to alter an already
// existing config in a way that doesn't require recreating it completely, and
// it would be complex and expensive to maintain class invariants.
// The CPU (processor) numbers always correspond to the actual numbering used
// by the system. The NUMA node numbers MAY NOT correspond to the system's
// numbering of the NUMA nodes. In particular, by default, if the processor has
// non-uniform cache access within a NUMA node (i.e., a non-unified L3 cache structure),
// then L3 domains within a system NUMA node will be used to subdivide it
// into multiple logical NUMA nodes in the config. Additionally, empty nodes may
// be removed, or the user may create custom nodes.
//
// As a special case, when performing system-wide replication of read-only data
// (i.e., LazyNumaReplicatedSystemWide), the system NUMA node is used, rather than
// custom or L3-aware nodes. See that class's get_discriminator() function.
//
// It is guaranteed that NUMA nodes are NOT empty: every node exposed by NumaConfig
// has at least one processor assigned.
//
// We use startup affinities so as not to modify its own behaviour in time.
//
// Since Stockfish doesn't support exceptions all places where an exception
// should be thrown are replaced by std::exit.
class NumaConfig {
   public:
    NumaConfig();

    // This function gets a NumaConfig based on the system's provided information.
    // The available policies are documented above.
    static NumaConfig from_system(const NumaAutoPolicy& policy,
                                  bool respectProcessAffinity = true);

    // ':'-separated numa nodes
    // ','-separated cpu indices
    // supports "first-last" range syntax for cpu indices
    // For example "0-15,128-143:16-31,144-159:32-47,160-175:48-63,176-191"
    static NumaConfig from_string(const std::string& s);

    NumaConfig(const NumaConfig&)            = delete;
    NumaConfig(NumaConfig&&)                 = default;
    NumaConfig& operator=(const NumaConfig&) = delete;
    NumaConfig& operator=(NumaConfig&&)      = default;

    bool is_cpu_assigned(CpuIndex n) const { return nodeByCpu.count(n) == 1; }

    NumaIndex num_numa_nodes() const { return nodes.size(); }

    CpuIndex num_cpus_in_numa_node(NumaIndex n) const {
        assert(n < nodes.size());
        return nodes[n].size();
    }

    CpuIndex num_cpus() const { return nodeByCpu.size(); }

    bool requires_memory_replication() const { return customAffinity || nodes.size() > 1; }

    std::string to_string() const;

    bool suggests_binding_threads(CpuIndex numThreads) const;

    std::vector<NumaIndex> distribute_threads_among_numa_nodes(CpuIndex numThreads) const;

    NumaReplicatedAccessToken bind_current_thread_to_numa_node(NumaIndex n) const;

    template<typename FuncT>
    void execute_on_numa_node(NumaIndex n, FuncT&& f) const {
        std::thread th([this, &f, n]() {
            bind_current_thread_to_numa_node(n);
            std::forward<FuncT>(f)();
        });

        th.join();
    }

    std::vector<std::set<CpuIndex>> nodes;
    std::map<CpuIndex, NumaIndex>   nodeByCpu;

   private:
    CpuIndex highestCpuIndex;

    bool customAffinity;

    static NumaConfig empty() { return NumaConfig(EmptyNodeTag{}); }

    struct EmptyNodeTag {};

    NumaConfig(EmptyNodeTag) :
        highestCpuIndex(0),
        customAffinity(false) {}

    void remove_empty_numa_nodes();

    // Returns true if successful
    // Returns false if failed, i.e. when the cpu is already present
    //                          strong guarantee, the structure remains unmodified
    bool add_cpu_to_node(NumaIndex n, CpuIndex c);

    // Returns true if successful
    // Returns false if failed, i.e. when any of the cpus is already present
    //                          strong guarantee, the structure remains unmodified
    bool add_cpu_range_to_node(NumaIndex n, CpuIndex cfirst, CpuIndex clast);

    static std::vector<size_t> indices_from_shortened_string(const std::string& s);

    // This function queries the system for the mapping of processors to NUMA nodes.
    // On Linux we read from standardized kernel sysfs, with a fallback to single NUMA
    // node. On Windows we utilize GetNumaProcessorNodeEx, which has its quirks, see
    // comment for Windows implementation of get_process_affinity.
    static NumaConfig from_system_numa(bool respectProcessAffinity, std::function<bool(CpuIndex)> is_cpu_allowed);

    static std::optional<NumaConfig> try_get_l3_aware_config(
      bool respectProcessAffinity, size_t bundleSize, std::function<bool(CpuIndex)> is_cpu_allowed);


    static NumaConfig from_l3_info(std::vector<L3Domain>&& domains, size_t bundleSize);
};

class NumaReplicationContext;

// Instances of this class are tracked by the NumaReplicationContext instance.
// NumaReplicationContext informs all tracked instances when NUMA configuration changes.
class NumaReplicatedBase {
   public:
    NumaReplicatedBase(NumaReplicationContext& ctx);

    NumaReplicatedBase(const NumaReplicatedBase&) = delete;
    NumaReplicatedBase(NumaReplicatedBase&& other) noexcept;

    NumaReplicatedBase& operator=(const NumaReplicatedBase&) = delete;
    NumaReplicatedBase& operator=(NumaReplicatedBase&& other) noexcept;

    virtual void on_numa_config_changed() = 0;
    virtual ~NumaReplicatedBase();

    const NumaConfig& get_numa_config() const;

   private:
    NumaReplicationContext* context;
};

// We force boxing with a unique_ptr. If this becomes an issue due to added
// indirection we may need to add an option for a custom boxing type. When the
// NUMA config changes the value stored at the index 0 is replicated to other nodes.
template<typename T>
class NumaReplicated: public NumaReplicatedBase {
   public:
    using ReplicatorFuncType = std::function<T(const T&)>;

    NumaReplicated(NumaReplicationContext& ctx) :
        NumaReplicatedBase(ctx) {
        replicate_from(T{});
    }

    NumaReplicated(NumaReplicationContext& ctx, T&& source) :
        NumaReplicatedBase(ctx) {
        replicate_from(std::move(source));
    }

    NumaReplicated(const NumaReplicated&) = delete;
    NumaReplicated(NumaReplicated&& other) noexcept :
        NumaReplicatedBase(std::move(other)),
        instances(std::exchange(other.instances, {})) {}

    NumaReplicated& operator=(const NumaReplicated&) = delete;
    NumaReplicated& operator=(NumaReplicated&& other) noexcept {
        NumaReplicatedBase::operator=(*this, std::move(other));
        instances = std::exchange(other.instances, {});

        return *this;
    }

    NumaReplicated& operator=(T&& source) {
        replicate_from(std::move(source));

        return *this;
    }

    ~NumaReplicated() override = default;

    const T& operator[](NumaReplicatedAccessToken token) const {
        assert(token.get_numa_index() < instances.size());
        return *(instances[token.get_numa_index()]);
    }

    const T& operator*() const { return *(instances[0]); }

    const T* operator->() const { return instances[0].get(); }

    template<typename FuncT>
    void modify_and_replicate(FuncT&& f) {
        auto source = std::move(instances[0]);
        std::forward<FuncT>(f)(*source);
        replicate_from(std::move(*source));
    }

    void on_numa_config_changed() override {
        // Use the first one as the source. It doesn't matter which one we use,
        // because they all must be identical, but the first one is guaranteed to exist.
        auto source = std::move(instances[0]);
        replicate_from(std::move(*source));
    }

   private:
    std::vector<std::unique_ptr<T>> instances;

    void replicate_from(T&& source) {
        instances.clear();

        const NumaConfig& cfg = get_numa_config();
        if (cfg.requires_memory_replication())
        {
            for (NumaIndex n = 0; n < cfg.num_numa_nodes(); ++n)
            {
                cfg.execute_on_numa_node(
                  n, [this, &source]() { instances.emplace_back(std::make_unique<T>(source)); });
            }
        }
        else
        {
            assert(cfg.num_numa_nodes() == 1);
            // We take advantage of the fact that replication is not required
            // and reuse the source value, avoiding one copy operation.
            instances.emplace_back(std::make_unique<T>(std::move(source)));
        }
    }
};

// We force boxing with a unique_ptr. If this becomes an issue due to added
// indirection we may need to add an option for a custom boxing type.
template<typename T>
class LazyNumaReplicated: public NumaReplicatedBase {
   public:
    using ReplicatorFuncType = std::function<T(const T&)>;

    LazyNumaReplicated(NumaReplicationContext& ctx) :
        NumaReplicatedBase(ctx) {
        prepare_replicate_from(T{});
    }

    LazyNumaReplicated(NumaReplicationContext& ctx, T&& source) :
        NumaReplicatedBase(ctx) {
        prepare_replicate_from(std::move(source));
    }

    LazyNumaReplicated(const LazyNumaReplicated&) = delete;
    LazyNumaReplicated(LazyNumaReplicated&& other) noexcept :
        NumaReplicatedBase(std::move(other)),
        instances(std::exchange(other.instances, {})) {}

    LazyNumaReplicated& operator=(const LazyNumaReplicated&) = delete;
    LazyNumaReplicated& operator=(LazyNumaReplicated&& other) noexcept {
        NumaReplicatedBase::operator=(*this, std::move(other));
        instances = std::exchange(other.instances, {});

        return *this;
    }

    LazyNumaReplicated& operator=(T&& source) {
        prepare_replicate_from(std::move(source));

        return *this;
    }

    ~LazyNumaReplicated() override = default;

    const T& operator[](NumaReplicatedAccessToken token) const {
        assert(token.get_numa_index() < instances.size());
        ensure_present(token.get_numa_index());
        return *(instances[token.get_numa_index()]);
    }

    const T& operator*() const { return *(instances[0]); }

    const T* operator->() const { return instances[0].get(); }

    template<typename FuncT>
    void modify_and_replicate(FuncT&& f) {
        auto source = std::move(instances[0]);
        std::forward<FuncT>(f)(*source);
        prepare_replicate_from(std::move(*source));
    }

    void on_numa_config_changed() override {
        // Use the first one as the source. It doesn't matter which one we use,
        // because they all must be identical, but the first one is guaranteed to exist.
        auto source = std::move(instances[0]);
        prepare_replicate_from(std::move(*source));
    }

   private:
    mutable std::vector<std::unique_ptr<T>> instances;
    mutable std::mutex                      mutex;

    void ensure_present(NumaIndex idx) const {
        assert(idx < instances.size());

        if (instances[idx] != nullptr)
            return;

        assert(idx != 0);

        std::unique_lock<std::mutex> lock(mutex);
        // Check again for races.
        if (instances[idx] != nullptr)
            return;

        const NumaConfig& cfg = get_numa_config();
        cfg.execute_on_numa_node(
          idx, [this, idx]() { instances[idx] = std::make_unique<T>(*instances[0]); });
    }

    void prepare_replicate_from(T&& source) {
        instances.clear();

        const NumaConfig& cfg = get_numa_config();
        if (cfg.requires_memory_replication())
        {
            assert(cfg.num_numa_nodes() > 0);

            // We just need to make sure the first instance is there.
            // Note that we cannot move here as we need to reallocate the data
            // on the correct NUMA node.
            cfg.execute_on_numa_node(
              0, [this, &source]() { instances.emplace_back(std::make_unique<T>(source)); });

            // Prepare others for lazy init.
            instances.resize(cfg.num_numa_nodes());
        }
        else
        {
            assert(cfg.num_numa_nodes() == 1);
            // We take advantage of the fact that replication is not required
            // and reuse the source value, avoiding one copy operation.
            instances.emplace_back(std::make_unique<T>(std::move(source)));
        }
    }
};

// Utilizes shared memory.
template<typename T>
class LazyNumaReplicatedSystemWide: public NumaReplicatedBase {
   public:
    using ReplicatorFuncType = std::function<T(const T&)>;

    LazyNumaReplicatedSystemWide(NumaReplicationContext& ctx) :
        NumaReplicatedBase(ctx) {
        prepare_replicate_from(std::make_unique<T>());
    }

    LazyNumaReplicatedSystemWide(NumaReplicationContext& ctx, std::unique_ptr<T>&& source) :
        NumaReplicatedBase(ctx) {
        prepare_replicate_from(std::move(source));
    }

    LazyNumaReplicatedSystemWide(const LazyNumaReplicatedSystemWide&) = delete;
    LazyNumaReplicatedSystemWide(LazyNumaReplicatedSystemWide&& other) noexcept :
        NumaReplicatedBase(std::move(other)),
        instances(std::exchange(other.instances, {})) {}

    LazyNumaReplicatedSystemWide& operator=(const LazyNumaReplicatedSystemWide&) = delete;
    LazyNumaReplicatedSystemWide& operator=(LazyNumaReplicatedSystemWide&& other) noexcept {
        NumaReplicatedBase::operator=(*this, std::move(other));
        instances = std::exchange(other.instances, {});

        return *this;
    }

    LazyNumaReplicatedSystemWide& operator=(std::unique_ptr<T>&& source) {
        prepare_replicate_from(std::move(source));

        return *this;
    }

    ~LazyNumaReplicatedSystemWide() override = default;

    const T& operator[](NumaReplicatedAccessToken token) const {
        assert(token.get_numa_index() < instances.size());
        ensure_present(token.get_numa_index());
        return *(instances[token.get_numa_index()]);
    }

    const T& operator*() const { return *(instances[0]); }

    const T* operator->() const { return &*instances[0]; }

    std::vector<std::pair<SystemWideSharedConstantAllocationStatus, std::optional<std::string>>>
    get_status_and_errors() const {
        std::vector<std::pair<SystemWideSharedConstantAllocationStatus, std::optional<std::string>>>
          status;
        status.reserve(instances.size());

        for (const auto& instance : instances)
        {
            status.emplace_back(instance.get_status(), instance.get_error_message());
        }

        return status;
    }

    template<typename FuncT>
    void modify_and_replicate(FuncT&& f) {
        auto source = std::make_unique<T>(*instances[0]);
        std::forward<FuncT>(f)(*source);
        prepare_replicate_from(std::move(source));
    }

    void on_numa_config_changed() override {
        // Use the first one as the source. It doesn't matter which one we use,
        // because they all must be identical, but the first one is guaranteed to exist.
        auto source = std::make_unique<T>(*instances[0]);
        prepare_replicate_from(std::move(source));
    }

   private:
    mutable std::vector<SystemWideSharedConstant<T>> instances;
    mutable std::mutex                               mutex;

    std::size_t get_discriminator(NumaIndex idx) const {
        const NumaConfig& cfg     = get_numa_config();
        const NumaConfig& cfg_sys = NumaConfig::from_system(SystemNumaPolicy{}, false);
        // as a discriminator, locate the hardware/system numadomain this cpuindex belongs to
        CpuIndex    cpu     = *cfg.nodes[idx].begin();  // get a CpuIndex from NumaIndex
        NumaIndex   sys_idx = cfg_sys.is_cpu_assigned(cpu) ? cfg_sys.nodeByCpu.at(cpu) : 0;
        std::string s       = cfg_sys.to_string() + "$" + std::to_string(sys_idx);
        return static_cast<std::size_t>(hash_string(s));
    }

    void ensure_present(NumaIndex idx) const {
        assert(idx < instances.size());

        if (instances[idx] != nullptr)
            return;

        assert(idx != 0);

        std::unique_lock<std::mutex> lock(mutex);
        // Check again for races.
        if (instances[idx] != nullptr)
            return;

        const NumaConfig& cfg = get_numa_config();
        cfg.execute_on_numa_node(idx, [this, idx]() {
            instances[idx] = SystemWideSharedConstant<T>(*instances[0], get_discriminator(idx));
        });
    }

    void prepare_replicate_from(std::unique_ptr<T>&& source) {
        instances.clear();

        const NumaConfig& cfg = get_numa_config();
        // We just need to make sure the first instance is there.
        // Note that we cannot move here as we need to reallocate the data
        // on the correct NUMA node.
        // Even in the case of a single NUMA node we have to copy since it's shared memory.
        if (cfg.requires_memory_replication())
        {
            assert(cfg.num_numa_nodes() > 0);

            cfg.execute_on_numa_node(0, [this, &source]() {
                instances.emplace_back(SystemWideSharedConstant<T>(*source, get_discriminator(0)));
            });

            // Prepare others for lazy init.
            instances.resize(cfg.num_numa_nodes());
        }
        else
        {
            assert(cfg.num_numa_nodes() == 1);
            instances.emplace_back(SystemWideSharedConstant<T>(*source, get_discriminator(0)));
        }
    }
};

class NumaReplicationContext {
   public:
    NumaReplicationContext(NumaConfig&& cfg) :
        config(std::move(cfg)) {}

    NumaReplicationContext(const NumaReplicationContext&) = delete;
    NumaReplicationContext(NumaReplicationContext&&)      = delete;

    NumaReplicationContext& operator=(const NumaReplicationContext&) = delete;
    NumaReplicationContext& operator=(NumaReplicationContext&&)      = delete;

    ~NumaReplicationContext() {
        // The context must outlive replicated objects
        if (!trackedReplicatedObjects.empty())
            std::exit(EXIT_FAILURE);
    }

    void attach(NumaReplicatedBase* obj) {
        assert(trackedReplicatedObjects.count(obj) == 0);
        trackedReplicatedObjects.insert(obj);
    }

    void detach(NumaReplicatedBase* obj) {
        assert(trackedReplicatedObjects.count(obj) == 1);
        trackedReplicatedObjects.erase(obj);
    }

    // oldObj may be invalid at this point
    void move_attached([[maybe_unused]] NumaReplicatedBase* oldObj, NumaReplicatedBase* newObj) {
        assert(trackedReplicatedObjects.count(oldObj) == 1);
        assert(trackedReplicatedObjects.count(newObj) == 0);
        trackedReplicatedObjects.erase(oldObj);
        trackedReplicatedObjects.insert(newObj);
    }

    void set_numa_config(NumaConfig&& cfg) {
        config = std::move(cfg);
        for (auto&& obj : trackedReplicatedObjects)
            obj->on_numa_config_changed();
    }

    const NumaConfig& get_numa_config() const { return config; }

   private:
    NumaConfig config;

    // std::set uses std::less by default, which is required for pointer comparison
    std::set<NumaReplicatedBase*> trackedReplicatedObjects;
};

}  // namespace Stockfish


#endif  // #ifndef NUMA_H_INCLUDED
