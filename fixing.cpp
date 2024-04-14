#include "newthreadpool.h"

#include <iostream>





int main(){
    ThreadPool *pool = ThreadPool::getInstance();

    pool->start();

    const size_t thread_count = pool->numThreads();

    std::cout << thread_count << "\n";

}