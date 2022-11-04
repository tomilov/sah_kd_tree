#include <utils/log.hpp>

#include <atomic>
#include <iostream>
#include <string_view>

namespace utils
{
struct LoggableBase::Impl
{
    std::atomic<LogLevel> logLevel = LogLevel::Debug;
};

LoggableBase::LoggableBase() = default;
LoggableBase::~LoggableBase() = default;

void LoggableBase::setLogLevel(LogLevel logLevel)
{
    impl_->logLevel = logLevel;
}

bool LoggableBase::checkLogLevel(LogLevel logLevel) const
{
    return logLevel >= impl_->logLevel;
}

void LoggableBase::log(std::string_view message, LogLevel logLevel) const
{
    if (!checkLogLevel(logLevel)) {
        return;
    }
    switch (logLevel) {
    case LogLevel::Debug: {
        std::cout << message << std::endl;
        break;
    }
    case LogLevel::Info: {
        std::cout << message << std::endl;
        break;
    }
    case LogLevel::Warning: {
        std::clog << message << std::endl;
        break;
    }
    case LogLevel::Critical: {
        std::cerr << message << std::endl;
        break;
    }
    }
}

}  // namespace utils
