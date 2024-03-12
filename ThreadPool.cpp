#include "ThreadPool.h"

ThreadPool ThreadPool::instance_;

thread_local size_t ThreadPool::thread_serial_number = 0;