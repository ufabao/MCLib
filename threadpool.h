#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <queue>



class join_threads{
public:
    explicit join_threads(std::vector<std::thread>& threads_) : threads(threads_) {}

    ~join_threads() {
        for(unsigned long i = 0; i < threads.size(); ++i){
            if(threads[i].joinable()) threads[i].join();
        }
    }

private:
    std::vector<std::thread>& threads;
};


template <typename T>
class thread_safe_queue{
public:

    thread_safe_queue() {}
    
    void push(T&& t){
        std::lock_guard<std::mutex> lk(mymutex);
        myqueue.push(std::move(t));
        mycv.notify_one();
    }

    bool try_pop(T& t){
        std::lock_guard<std::mutex> lk(mymutex);
        if(myqueue.empty()) return false;
        t = std::move(myqueue.front());
        myqueue.pop();
        return true;
    }

    void wait_and_pop(T& t){
        std::unique_lock<std::mutex> lk(mymutex);
        while(myqueue.empty()){
            mycv.wait(lk);
        }
        t = std::move(myqueue.front());
        myqueue.pop();
    }

    thread_safe_queue(thread_safe_queue&) = delete;
    thread_safe_queue& operator=(thread_safe_queue&) = delete;

    thread_safe_queue(thread_safe_queue&& rhs){
        std::lock_guard<std::mutex> lk(rhs.mymutex);
        myqueue = std::move(rhs.myqueue);
    }

private:
    std::queue<T>               myqueue;
    mutable std::mutex          mymutex;
    std::condition_variable     mycv;
};




class function_wrapper{
public:
    template <typename F>
    function_wrapper(F&& f) : impl(new impl_type<F>(std::move(f))) {}

    void operator()() { impl->call(); }

    function_wrapper() = default;
    function_wrapper(function_wrapper&& other) : impl(std::move(other.impl)) {} 

    function_wrapper& operator=(function_wrapper&& other){
        if(this == &other) return *this;
        impl = std::move(other.impl);
        return *this;
    }

    function_wrapper(const function_wrapper&) = delete;
    function_wrapper(function_wrapper&) = delete;
    function_wrapper& operator=(const function_wrapper&) = delete;

private:
    struct impl_base{
        virtual void call() = 0;
        virtual ~impl_base() {}
    };

    std::unique_ptr<impl_base> impl;
    template <typename F>
    struct impl_type : impl_base {
        F f;
        impl_type(F&& f_) : f(std::move(f)) {}
        void call() { f(); }
    }; 
};




class thread_pool{
public:
    thread_pool() : done(false), joiner(threads) {
        unsigned const thread_count = std::thread::hardware_concurrency();
        try{
            for(unsigned i = 0; i < thread_count; ++i){
                threads.push_back(
                    std::thread(&thread_pool::worker_thread, this));
            }
        }
        catch(...){
            done = true;
            throw;
        }
    }

    ~thread_pool() {
        done = true;
    }

    template <typename FunctionType>
    std::future<typename std::result_of<FunctionType()>::type>
    submit(FunctionType f){
        using result_type =  typename std::result_of<FunctionType()>::type;
        std::packaged_task<result_type()> task(std::move(f));
        std::future<result_type> res(task.get_future());
        work_queue.push(std::move(task));
        return res;
    }

    const size_t numThreads() const {
        return threads.size();
    }

private:
    std::atomic_bool done;
    thread_safe_queue<function_wrapper> work_queue;
    std::vector<std::thread> threads;
    join_threads joiner;

    void worker_thread(){
        while(!done){
            function_wrapper task;
            if(work_queue.try_pop(task)){
                task();
            }
            else{
                std::this_thread::yield();
            }
        }
    }
};

