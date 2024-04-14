#pragma once
#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <future>


using namespace std::chrono_literals;

// This is a thread safe queue that will be the threadpools SPMC queue.
template <typename T>
class ThreadSafeQueue{
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cond_var_;
    bool interrupt_{false};

public:
    ThreadSafeQueue() : interrupt_(false) {}

    ~ThreadSafeQueue() {
        interrupt();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lk(mutex_);
        return queue_.empty();
    }

    void push(T t) {
        std::lock_guard<std::mutex> lk(mutex_);
        queue_.push(std::move(t));
        cond_var_.notify_one();
    }

    bool try_pop(T& t){
        std::lock_guard<std::mutex> lk(mutex_);
        if(queue_.empty()) return false;

        // need to use moves because copy assignment is deleted for packaged_tasks
        t = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    void pop(T& t) {
        std::unique_lock<std::mutex> lk(mutex_);
        while(!interrupt_ && queue_.empty()) cond_var_.wait(lk);
        t = std::move(queue_.front());
        queue_.pop();
    }

    void interrupt() {
        {
            std::lock_guard<std::mutex> lk(mutex_);
            interrupt_ = true;
        }

        cond_var_.notify_all();
    }

    void reset_interrupt() {
        interrupt_ = false;
    }

    void clear() {
        std::queue<T> empty;
        std::swap(queue_, empty);
    }
};

// The parallel algorithm will use a threadpool as our executor.
class ThreadPool{
    //singleton pattern
    static ThreadPool instance_;
    ThreadPool(): active_(false), interrupt_(false) {}

    bool active_;
    bool interrupt_;

    ThreadSafeQueue<std::packaged_task<bool(void)>> queue_;
    std::vector<std::thread> threads_;


    void thread_function(const size_t n){
        thread_serial_number = n;
        std::packaged_task<bool(void)> task;
        while(!interrupt_){
            queue_.pop(task);
            if(!interrupt_) task();
        }
    }

public:
    static thread_local size_t thread_serial_number;

    static ThreadPool* get_instance() {return &instance_;}

    ThreadPool(const ThreadPool& rhs) = delete;
    ThreadPool(ThreadPool&& rhs) = delete;
    ThreadPool& operator=(const ThreadPool& rhs) = delete;
    ThreadPool& operator=(ThreadPool&& rhs) = delete;

    void start(const size_t num_threads = std::thread::hardware_concurrency() - 1){
        if(!active_){
            threads_.reserve(num_threads);
            for(auto i = 0; i < num_threads; ++i){
                threads_.push_back(std::thread(&ThreadPool::thread_function, this, i + 1));
            }
            active_ = true;
        }
    }

    ~ThreadPool(){
        stop();
    };

    size_t number_of_threads() const {return threads_.size();}

    static size_t thread_number() {return thread_serial_number;}

    // stop() is called from the destructor, so just cleans up the threadpool and queue on program exit.
    void stop(){
        if(active_){
            interrupt_ = true;
            queue_.interrupt();

            std::for_each(threads_.begin(), threads_.end(), std::mem_fn(&std::thread::join));
            threads_.clear();
            queue_.clear();
            queue_.reset_interrupt();
            active_ = false;
            interrupt_ = false;
        }
    }

    // takes a lambda f, converts it into a promise which is pushed onto the queue, and returns a future for that promise
    template <typename F>
    std::future<bool> spawn_task(F f){
        std::packaged_task<bool(void)> prom(std::move(f));
        std::future<bool> future_ = prom.get_future();
        queue_.push(std::move(prom));
        return future_;
    }


    // once the main thread is done populating the queue it will steal some work. This feature is currently not working
    bool active_wait(const std::future<bool>& fut){
        std::packaged_task<bool(void)> task;
        bool res{false};

        // doing this wait on fut prevents blocking
        while(fut.wait_for(0s) != std::future_status::ready){
            if(queue_.try_pop(task)){
                task();
                res = true;
            }
            else{
                fut.wait();
            }
        }

        return res;
    }
};
