// CPU lifecycle regression. The included methods and RPC declarations are
// extracted verbatim from production by test_loader_close.py. Only GPU/IO
// resource operations are replaced; queues and the worker executor are real.
#include <any>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <instant_tensor/function_executor.hpp>

namespace instanttensor {
using std::unique_ptr;
using chunk_id_t = int64_t;
#include "production_rpc.inc"
inline bool _env_debug() { return false; }
template<typename... Args> void debug_log(const char*, Args...) {}

class Loader {
public:
    unique_ptr<SPSCQueue<RPCRequest>> input_queue = std::make_unique<SPSCQueue<RPCRequest>>();
    unique_ptr<SPSCQueue<RPCResponse>> output_queue = std::make_unique<SPSCQueue<RPCResponse>>();
    std::atomic<bool> stop{false};
    std::atomic<size_t> io_depth_sum{0}, io_depth_sample{1};
    SingleWorkerFunctionExecutor<> executor;
    struct Stats {
        std::array<double, 3> average_stage_counts() { return {0, 0, 0}; }
        double average_active_tasks() { return 0; }
    } cuda_stats, io_stats;
    std::exception_ptr fatal_error;
    void set_fatal_error(std::exception_ptr error);
    void rethrow_fatal_error() const;
    int world_size = 1;
    size_t current_tensor_index = 0;
    std::vector<int> tensors, chunks;
    bool pending = true;
    bool buffer_freed = false;
    bool file_closed = false;
    int steps = 0;
    int ordinary_requests = 0;
    void close(CloseArgs args);
    void run();
    void wait_step(int64_t) { assert(false); } // single-rank close never drains
    void destroy_threads() { executor.join(); }
    void destroy_buffer() { buffer_freed = true; }
    void close_file() { file_closed = true; }
    bool can_step() { return pending; }
    void try_step() {
        ++steps;
        if (pending) {
            executor.submit(7, [] {});
            executor.reap(7);
        }
    }
    std::any dispatch(const RPCRequest& request) {
        if (request.op == CLOSE) close(CloseArgs{});
        else ++ordinary_requests;
        return {};
    }
};
#include "production_loader_methods.inc"

class LoaderManager {
public:
    std::unordered_map<int, RPCHandle> loader_handle_to_queue;
#include "production_manager_result.inc"
};

template<typename Owner, typename Queue>
Owner response_owner(const std::shared_ptr<Queue>& queue) {
    if constexpr (std::is_pointer_v<Owner>) return queue.get();
    else return queue;
}

void close_with_prefetch(bool pending) {
    Loader loader;
    loader.pending = pending;
    loader.input_queue->push(RPCRequest{42, CLOSE, {}});
    loader.run();
    assert(loader.stop && loader.buffer_freed && loader.file_closed);
    assert(loader.steps == 0); // no access to stopped executors or freed buffers
    RPCResponse result;
    assert(loader.output_queue->try_pop(result) && result.id == 42);
    assert(!loader.output_queue->try_pop(result));
}

void response_after_loader_exit() {
    LoaderManager manager;
    using ResponseQueue = SPSCQueue<RPCResponse>;
    auto queue = std::make_shared<ResponseQueue>();
    std::weak_ptr<ResponseQueue> lifetime = queue;
    manager.loader_handle_to_queue.emplace(1, RPCHandle{
        0, nullptr, response_owner<decltype(RPCHandle::output_queue)>(queue)});
    // Schedule the caller after the detached producer has finished. This is
    // legal: a CLOSE reply must survive even if the loader exits immediately.
    std::thread producer([queue = std::move(queue)] {
        queue->push(RPCResponse{42, std::any(123)});
    });
    producer.join();
    // Do not let an expired weak_ptr keep the destroyed object storage around;
    // ASan should diagnose the original raw-pointer use after free.
    if (lifetime.expired()) lifetime.reset();
    auto response = manager.get_result(1, 42);
    assert(std::any_cast<int>(response) == 123);
    assert(!lifetime.expired());
    manager.loader_handle_to_queue.erase(1);
    assert(lifetime.expired()); // no leaked response queue after consumption
}
}

int main(int argc, char** argv) {
    assert(argc == 2);
    try {
        const std::string which = argv[1];
        if (which == "pending") instanttensor::close_with_prefetch(true);
        else if (which == "exhausted") instanttensor::close_with_prefetch(false);
        else if (which == "reply") instanttensor::response_after_loader_exit();
        else return 2;
    } catch (const std::exception& e) {
        std::cerr << "Production lifecycle raised: " << e.what() << '\n';
        return 1;
    }
}
