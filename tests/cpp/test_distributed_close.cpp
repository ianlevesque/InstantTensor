// Compile the production close/step/wait methods against a CPU collective
// rendezvous and real executor. Skew is explicit, never a scheduling accident.
#include <any>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <instant_tensor/function_executor.hpp>

namespace instanttensor {
using chunk_id_t = int64_t;
struct CloseArgs {};
struct TensorMetadate { chunk_id_t last_chunk_id, prefetch_chunk_id; };
struct ChunkRequest { SingleWorkerFunctionExecutor<>* executor; int wait_handle; };
struct Chunk { ChunkRequest request; };
inline bool _env_debug() { return false; }
template<typename... Args> void debug_log(const char*, Args...) {}
template<typename E> [[noreturn]] void print_and_throw(E e) { throw e; }

struct Collectives {
    std::mutex mutex;
    std::condition_variable cv;
    std::array<int, 2> submitted{0, 0};
    void submit(int rank, int chunk) {
        std::lock_guard lock(mutex);
        assert(submitted[rank] == chunk);
        ++submitted[rank];
        cv.notify_all();
    }
    void wait(int chunk) {
        std::unique_lock lock(mutex);
        if (!cv.wait_for(lock, std::chrono::seconds(2), [&] {
            return submitted[0] > chunk && submitted[1] > chunk;
        })) {
            // An unmatched native collective would never signal its CUDA
            // event. Bound the test and retain the exact submitted counts.
            std::cerr << "unmatched collective " << chunk << ": rank counts "
                      << submitted[0] << ", " << submitted[1] << '\n';
            std::abort();
        }
    }
};

class Loader {
public:
    std::atomic<bool> stop{false};
    std::atomic<size_t> io_depth_sum{0}, io_depth_sample{1};
    std::atomic<chunk_id_t> chunk_read{-1}, chunk_reading{-1};
    int world_size = 2;
    int rank;
    size_t current_tensor_index = 0;
    // This layout ends at selected chunk 7. Unselected source-file bytes are
    // deliberately absent from the plan and must never be submitted.
    std::vector<TensorMetadate> tensors{{0, 2}, {2, 4}, {4, 6}, {7, 7}};
    std::vector<Chunk> chunks = std::vector<Chunk>(8);
    Collectives* collectives;
    SingleWorkerFunctionExecutor<> executor;
    bool threads_joined = false, buffer_freed = false, file_closed = false;
    bool fail_post = false, fail_close = false;
    Loader(int rank, Collectives* collectives) : rank(rank), collectives(collectives) {}
    void close(CloseArgs);
    void poll_read_chunk();
    void wait_read_chunk(chunk_id_t);
    void step();
    bool can_step();
    void wait_step(chunk_id_t);
    void post_read_chunk() {
        if (fail_post) throw std::runtime_error("injected selected IO failure");
        const auto chunk = ++chunk_reading;
        assert(chunk >= 0 && static_cast<size_t>(chunk) < chunks.size());
        // Match production's bounded window reuse before submitting IO.
        if (chunk >= 3) wait_read_chunk(chunk - 3);
        if (world_size > 1) collectives->submit(rank, chunk);
        executor.submit(chunk, [this, chunk] {
            if (world_size > 1) collectives->wait(chunk);
        });
        chunks[chunk].request = ChunkRequest{&executor, static_cast<int>(chunk)};
    }
    void destroy_threads() {
        if (fail_close) throw std::runtime_error("injected close failure");
        executor.join();
        threads_joined = true;
    }
    void destroy_buffer() {
        assert(threads_joined);
        buffer_freed = true;
    }
    void close_file() { assert(buffer_freed); file_closed = true; }
};
#include "production_distributed_methods.inc"

void check_closed(const Loader& loader) {
    assert(loader.stop && loader.threads_joined && loader.buffer_freed && loader.file_closed);
}

void close_pair(bool different_position, bool complete) {
    Collectives collectives;
    Loader early(0, &collectives), late(1, &collectives);
    const int common = complete ? 8 : different_position ? 3 : 1;
    for (int i = 0; i < common; ++i) {
        early.post_read_chunk();
        late.post_read_chunk();
    }
    early.wait_read_chunk(common - 1);
    late.wait_read_chunk(common - 1);
    if (!complete) late.post_read_chunk(); // one unmatched prefetch on rank 1
    early.current_tensor_index = complete ? 3 : 0;
    late.current_tensor_index = complete ? 3 : different_position ? 2 : 0;
    std::thread a([&] { early.close({}); });
    std::thread b([&] { late.close({}); });
    a.join();
    b.join();
    check_closed(early);
    check_closed(late);
    assert(collectives.submitted[0] == 8 && collectives.submitted[1] == 8);
    assert(early.chunk_read == 7 && late.chunk_read == 7);
}

void close_without_drain(const std::string& which) {
    Collectives collectives;
    Loader loader(0, &collectives);
    if (which == "empty") loader.tensors.clear();
    else if (which == "empty-plan") loader.chunks.clear();
    else loader.world_size = 1;
    loader.close({});
    check_closed(loader);
    assert(loader.chunk_reading == -1);
    assert(collectives.submitted[0] == 0 && collectives.submitted[1] == 0);
}

void failure(bool in_drain) {
    Collectives collectives;
    Loader loader(0, &collectives);
    loader.fail_post = in_drain;
    loader.fail_close = !in_drain;
    if (!in_drain) loader.world_size = 1;
    bool caught = false;
    try { loader.close({}); }
    catch (const std::runtime_error& e) {
        caught = true;
        assert(std::string(e.what()) == (in_drain ?
            "injected selected IO failure" : "injected close failure"));
    }
    assert(caught);
    // Never acknowledge success or free storage whose work did not drain.
    // Production run_loader retains its existing terminal failure behavior;
    // this test does not claim communicator recovery after a native IO error.
    assert(!loader.threads_joined && !loader.buffer_freed && !loader.file_closed);
    assert(loader.stop == !in_drain);
}
}

int main(int argc, char** argv) {
    assert(argc == 2);
    const std::string which = argv[1];
    if (which == "same-position") instanttensor::close_pair(false, false);
    else if (which == "different-position") instanttensor::close_pair(true, false);
    else if (which == "complete") instanttensor::close_pair(true, true);
    else if (which == "empty" || which == "empty-plan" || which == "single")
        instanttensor::close_without_drain(which);
    else if (which == "drain-failure") instanttensor::failure(true);
    else if (which == "close-failure") instanttensor::failure(false);
    else if (which == "repeat") {
        for (int i = 0; i < 4; ++i) instanttensor::close_pair(i % 2, false);
    }
    else return 2;
}
