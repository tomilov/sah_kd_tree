#include <builder/builder.hpp>
#include <compute/compute.hpp>
#include <compute/make.hpp>
#include <scene_data/scene_data.hpp>
#include <scene_loader/scene_loader.hpp>
#include <soft_renderer/soft_renderer.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
#include <utils/pp.hpp>
#include <utils/scope_guard.hpp>

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <fmt/base.h>
#include <fmt/chrono.h>
#include <gli/convert.hpp>
#include <gli/save.hpp>
#include <gli/texture2d.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <spdlog/spdlog.h>

#include <QtCore/QDebug>
#include <QtCore/QDir>
#include <QtCore/QFileInfo>
#include <QtCore/QLoggingCategory>
#include <QtCore/QStandardPaths>
#include <QtCore/QString>
#include <QtCore/QtLogging>

#include <chrono>
#include <iterator>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace Qt::StringLiterals;
using namespace std::string_view_literals;

namespace
{
Q_DECLARE_LOGGING_CATEGORY(softRendererMain)
Q_LOGGING_CATEGORY(softRendererMain, "soft_renderer.main")

constexpr float kEmptinessFactor = 0.8f;
constexpr float kTraversalCost = 2.0f;
constexpr float kIntersectionCost = 1.0f;
constexpr uint32_t kMaxTreeDepth = 1000;

enum class ThrustDeviceSystem
{
    ThrustDeviceSystemDefault,
    ThrustDeviceSystemCPP,
    ThrustDeviceSystemOMP,
    ThrustDeviceSystemTBB,
    ThrustDeviceSystemCUDA,
};

scene_data::SceneDataPtr getScene(QString sceneFileName, glm::vec3 & sceneCenter, glm::float32 & mainDiagonal)
{
    scene_data::SceneData sceneData;
    QFileInfo sceneFileInfo{sceneFileName};
    if ((false)) {
        if (!scene_loader::load(sceneData, sceneFileInfo)) {
            qCDebug(softRendererMain).noquote() << u"Cannot load scene from file %1"_s.arg(sceneFileName);
            return {};
        }
    } else {
        auto cachePath = QStandardPaths::writableLocation(QStandardPaths::CacheLocation);
        if (!scene_loader::cachingLoad(sceneData, sceneFileInfo, cachePath.isEmpty() ? QDir::temp() : cachePath)) {
            qCDebug(softRendererMain).noquote() << u"Cannot load scene from file %1"_s.arg(sceneFileName);
            return {};
        }
    }
    sceneCenter = glm::mix(sceneData.aabb.min, sceneData.aabb.max, 0.5f);
    mainDiagonal = glm::distance(sceneData.aabb.min, sceneData.aabb.max);
    return std::make_shared<scene_data::SceneData>(std::move(sceneData));
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
std::unique_ptr<std::FILE, decltype(&std::fclose)> openFile(const char * filepath, std::FILE * stream)
{
#pragma GCC diagnostic pop
    if (filepath == "-"sv) {
        static constexpr decltype(&std::fclose) kNoop = [](std::FILE *) -> int
        {
            return 0;
        };
        return {stream, kNoop};
    }
    return {std::fopen(filepath, "wb"), std::fclose};
}

struct FPSCounter
{
    bool operator()(std::string & windowTitle, glm::float32 & dt, Uint64 updateIntervalMs = 500)
    {
        const Uint64 timeNow = SDL_GetPerformanceCounter();
        const Uint64 timeDelta = timeNow - std::exchange(timePrev, timeNow);

        dt = static_cast<glm::float32>(timeDelta) / static_cast<glm::float32>(frequency);

        frameCount++;
        titleTimer += timeDelta;
        if (titleTimer < frequency * updateIntervalMs / 1000) {
            return false;
        }
        const double fps = frameCount / (static_cast<double>(titleTimer) / static_cast<double>(frequency));
        const double frameTimeMs = static_cast<double>(timeDelta) / static_cast<double>(frequency) * 1000.0;
        frameCount = 0;
        titleTimer = 0;
        windowTitle.resize(0);
        fmt::format_to(std::back_inserter(windowTitle), "FPS: {:.1f} | Frame: {:.3f} ms", fps, frameTimeMs);
        return true;
    }

private:
    const Uint64 frequency = SDL_GetPerformanceFrequency();
    Uint64 timePrev = SDL_GetPerformanceCounter();
    Uint64 titleTimer = 0;
    int frameCount = 0;
};

}  // namespace

#define CALL_SDL(f, ...)                                                         \
    do                                                                           \
        if (!SDL_##f(__VA_ARGS__)) {                                             \
            SPDLOG_ERROR("SDL_" #f " failed: {}", SDL_GetError());               \
            throw std::runtime_error("Error: " STRINGIZE(SDL_##f(__VA_ARGS__))); \
        }                                                                        \
    while (false)

int main(int argc, char * argv[])
{
    glm::float32 mainDiagonal{-1.0f};
    glm::vec3 sceneCenter{0.0f};
    INVARIANT(argc > 1, "{}", argc);
    auto sceneData = getScene(QString::fromUtf8(argv[1]), sceneCenter, mainDiagonal);

    CALL_SDL(Init, SDL_INIT_VIDEO);
    utils::ScopeGuard sdlQuit{SDL_Quit};

    gli::extent2d::value_type width = 1024;
    gli::extent2d::value_type height = 768;
    constexpr SDL_WindowFlags kWindowFlags = SDL_WINDOW_RESIZABLE;
    std::unique_ptr<SDL_Window, decltype(&SDL_DestroyWindow)> window{SDL_CreateWindow(APPLICATION_NAME, utils::autoCast(width), utils::autoCast(height), kWindowFlags), &SDL_DestroyWindow};
    if (!window) {
        SPDLOG_ERROR("SDL_CreateWindow failed: {}", SDL_GetError());
        return EXIT_FAILURE;
    }
    CALL_SDL(SetWindowMinimumSize, window.get(), 16, 16);

    std::unique_ptr<SDL_Renderer, decltype(&SDL_DestroyRenderer)> renderer{SDL_CreateRenderer(window.get(), nullptr), &SDL_DestroyRenderer};
    if (!renderer) {
        SPDLOG_ERROR("SDL_CreateRenderer failed: {}", SDL_GetError());
        return EXIT_FAILURE;
    }

    const auto createTexture = [&renderer, &width, &height]() -> std::unique_ptr<SDL_Texture, decltype(&SDL_DestroyTexture)>
    {
        return {SDL_CreateTexture(renderer.get(), SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STREAMING, utils::autoCast(width), utils::autoCast(height)), &SDL_DestroyTexture};
    };
    auto texture = createTexture();
    if (!texture) {
        SPDLOG_ERROR("SDL_CreateTexture failed: {}", SDL_GetError());
        return EXIT_FAILURE;
    }

    const glm::vec4 kClearColor{0.0f, 0.0f, 0.0f, 1.0f};
    soft_renderer::SoftRenderer softRenderer{APPLICATION_NAME ""sv, kClearColor};

    const auto getTree = [cudaDevice = compute::makeCudaDevice(std::nullopt), sceneData = std::move(sceneData)](ThrustDeviceSystem thrustDeviceSystem) -> builder::TreePtr
    {
        const builder::Settings settings = {
            .emptinessFactor = kEmptinessFactor,
            .traversalCost = kTraversalCost,
            .intersectionCost = kIntersectionCost,
            .maxTreeDepth = kMaxTreeDepth,
        };
        const auto progress = [start = std::chrono::steady_clock::now()](size_t progressValue)
        {
            using namespace std::chrono_literals;
            if (start + 600s < std::chrono::steady_clock::now()) {
                INVARIANT(false, "{}", progressValue);
            }
            return false;
        };
        auto build = builder::getBuild(utils::autoCast(thrustDeviceSystem));
        if (!build) {
            return nullptr;
        }
        return build(settings, *cudaDevice, sceneData, progress);
    };
    std::optional<ThrustDeviceSystem> thrustDeviceSystem = ThrustDeviceSystem::ThrustDeviceSystemDefault;

    constexpr glm::float32 kCrossSceneAabbTime = 5.0f;
    constexpr glm::float32 kMouseSensetivity = 0.002f;
    glm::float32 yaw = 0.0f;
    glm::float32 pitch = 0.0f;

    bool keyW = false;
    bool keyA = false;
    bool keyS = false;
    bool keyD = false;
    bool keyE = false;
    bool keyQ = false;

    soft_renderer::FrameSettings frameSettings;
    {
        frameSettings.position = sceneCenter;
        // glm::quat orientation = glm::conjugate(glm::toQuat(glm::lookAt(position, glm::vec3{position.x, position.y, 0.0f}, glm::vec3{0.0f, -1.0f, 0.0f})));
        glm::vec3 eulerAngles{yaw, pitch, 0.0f};
        frameSettings.orientation = glm::quat{glm::radians(eulerAngles)};
    }

    const auto createTarget = [&width, &height]() -> gli::texture2d
    {
        return {soft_renderer::SoftRenderer::kTargetFormat, gli::extent2d{width, height}};
    };
    auto rgbaTarget = createTarget();

    using Clock = std::chrono::high_resolution_clock;
    auto getMillisecondsSinceLastCall = [start = Clock::now()]() mutable
    {
        auto now = Clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(now - std::exchange(start, now));
    };

    std::string windowTitle;
    std::string windowTitleFPS;
    std::string windowTitleBuildTime;
    const auto updateWindowTitle = [&]
    {
        windowTitle.resize(0);
        fmt::format_to(std::back_inserter(windowTitle), APPLICATION_NAME " | {} | {}", windowTitleFPS, windowTitleBuildTime);
        CALL_SDL(SetWindowTitle, window.get(), windowTitle.c_str());
    };
    FPSCounter fpsCounter;
    bool capturing = false;
    bool running = true;
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
            case SDL_EVENT_QUIT: {
                running = false;
                break;
            }
            case SDL_EVENT_WINDOW_RESIZED: {
                width = event.window.data1;
                height = event.window.data2;
                texture = createTexture();
                if (!texture) {
                    SPDLOG_ERROR("SDL_CreateTexture failed: {}", SDL_GetError());
                    return EXIT_FAILURE;
                }
                rgbaTarget = createTarget();
                break;
            }
            case SDL_EVENT_KEY_DOWN:
            case SDL_EVENT_KEY_UP: {
                const bool isPressed = (event.type == SDL_EVENT_KEY_DOWN);
                switch (event.key.scancode) {
                case SDL_SCANCODE_W: {
                    keyW = isPressed;
                    break;
                }
                case SDL_SCANCODE_A: {
                    keyA = isPressed;
                    break;
                }
                case SDL_SCANCODE_S: {
                    keyS = isPressed;
                    break;
                }
                case SDL_SCANCODE_D: {
                    keyD = isPressed;
                    break;
                }
                case SDL_SCANCODE_Q: {
                    keyQ = isPressed;
                    break;
                }
                case SDL_SCANCODE_E: {
                    keyE = isPressed;
                    break;
                }
                case SDL_SCANCODE_ESCAPE: {
                    if (isPressed) {
                        running = false;
                    }
                    break;
                }
                case SDL_SCANCODE_PRINTSCREEN: {
                    if (event.type == SDL_EVENT_KEY_UP) {
                        if (argc > 2) {
                            auto rgbTarget = gli::convert(rgbaTarget, gli::format::FORMAT_RGB8_UNORM_PACK8);
                            const std::string_view outputFilepath{argv[2]};
                            if ((outputFilepath == "-"sv) || outputFilepath.ends_with(".ppm"sv)) {
                                auto outputFile = openFile(argv[2], stdout);
                                if (!outputFile) {
                                    return EXIT_FAILURE;
                                }
                                fmt::println(outputFile.get(), "P6\n{} {}\n255", rgbTarget.extent().x, rgbTarget.extent().y);
                                const size_t writeSize = rgbTarget.size();
                                const size_t writtenSize = std::fwrite(rgbTarget.data(), 1, writeSize, outputFile.get());
                                if (writtenSize != writeSize) {
                                    SPDLOG_ERROR("{} != {}", writtenSize, writeSize);
                                    return EXIT_FAILURE;
                                }
                            } else {
                                if (!gli::save(rgbTarget, argv[2])) {
                                    SPDLOG_ERROR("gli::save({}) failed", argv[2]);
                                    return EXIT_FAILURE;
                                }
                            }
                            SPDLOG_ERROR("Screenshot saved to '{}'", argv[2]);
                        }
                    }
                    break;
                }
                case SDL_SCANCODE_1:
                case SDL_SCANCODE_2:
                case SDL_SCANCODE_3:
                case SDL_SCANCODE_4:
                case SDL_SCANCODE_5: {
                    if (isPressed && !event.key.repeat) {
                        thrustDeviceSystem = utils::safeCast<ThrustDeviceSystem>(event.key.scancode - SDL_SCANCODE_1);
                    }
                    break;
                }
                default: {
                    break;
                }
                }
                break;
            }
            case SDL_EVENT_MOUSE_MOTION: {
                if (capturing) {
                    yaw += event.motion.xrel * kMouseSensetivity;
                    pitch += event.motion.yrel * kMouseSensetivity;
                    constexpr auto kHalfPi = glm::half_pi<glm::float32>() - 0.01f;
                    pitch = glm::clamp(pitch, -kHalfPi, kHalfPi);
                }
                break;
            }
            case SDL_EVENT_MOUSE_BUTTON_DOWN: {
                if (event.button.button == SDL_BUTTON_LEFT) {
                    capturing = true;
                    CALL_SDL(SetWindowRelativeMouseMode, window.get(), true);
                }
                break;
            }
            case SDL_EVENT_MOUSE_BUTTON_UP: {
                if (event.button.button == SDL_BUTTON_LEFT) {
                    capturing = false;
                    CALL_SDL(SetWindowRelativeMouseMode, window.get(), false);
                }
                break;
            }
            case SDL_EVENT_WINDOW_FOCUS_LOST: {
                CALL_SDL(SetWindowRelativeMouseMode, window.get(), false);
                break;
            }
            case SDL_EVENT_WINDOW_FOCUS_GAINED: {
                if (capturing) {
                    if (event.button.button == SDL_BUTTON_LEFT) {
                        CALL_SDL(SetWindowRelativeMouseMode, window.get(), true);
                    } else {
                        capturing = false;
                    }
                }
                break;
            }
            default: {
                break;
            }
            }
        }

        glm::float32 dt;
        if (fpsCounter(windowTitleFPS, dt)) {
            updateWindowTitle();
        }

        const glm::quat qYaw = glm::angleAxis(yaw, glm::vec3{0.0f, 1.0f, 0.0f});
        const glm::quat qPitch = glm::angleAxis(pitch, glm::vec3{1.0f, 0.0f, 0.0f});
        frameSettings.orientation = glm::normalize(qYaw * qPitch);

        const glm::vec3 forward = frameSettings.orientation * glm::vec3{0.0f, 0.0f, 1.0f};
        const glm::vec3 right = frameSettings.orientation * glm::vec3{1.0f, 0.0f, 0.0f};
        const glm::vec3 up{0.0f, 1.0f, 0.0f};

        glm::vec3 moveDir{0.0f};
        if (keyW) {
            moveDir += forward;
        }
        if (keyS) {
            moveDir -= forward;
        }
        if (keyD) {
            moveDir += right;
        }
        if (keyA) {
            moveDir -= right;
        }
        if (keyE) {
            moveDir += up;
        }
        if (keyQ) {
            moveDir -= up;
        }

        auto getSpeedModifier = []
        {
            const SDL_Keymod kmod = SDL_GetModState();
            if (kmod & SDL_KMOD_SHIFT) {
                return 0.05f;
            }
            if (kmod & SDL_KMOD_CTRL) {
                return 5.0f;
            }
            return 1.0f;
        };
        if (glm::dot(moveDir, moveDir) > 0.0f) {
            moveDir = glm::normalize(moveDir);
        }
        frameSettings.position += moveDir * ((mainDiagonal / kCrossSceneAabbTime) * dt * getSpeedModifier());
        frameSettings.zFar = mainDiagonal + glm::distance(frameSettings.position, sceneCenter);
        frameSettings.zNear = frameSettings.zFar * std::numeric_limits<glm::float32>::epsilon() * 1000.0f;

        if (thrustDeviceSystem) {
            getMillisecondsSinceLastCall();
            if (auto tree = getTree(std::exchange(thrustDeviceSystem, std::nullopt).value())) {
                softRenderer.setTree(std::move(*tree));
            }
            windowTitleBuildTime.resize(0);
            fmt::format_to(std::back_inserter(windowTitleBuildTime), "Built in: {}", getMillisecondsSinceLastCall());
            updateWindowTitle();
        }
        if (softRenderer.hasTree()) {
            softRenderer.render(frameSettings, rgbaTarget);
        }

        uint8_t * pixels = nullptr;
        int sdlPitch = 0;
        if (SDL_LockTexture(texture.get(), nullptr, utils::autoCast(&pixels), &sdlPitch)) {
            utils::ScopeGuard sdlUnlockTexture{SDL_UnlockTexture, texture.get()};
            const uint8_t * srcData = utils::autoCast(rgbaTarget.data(0, 0, 0));
            const auto srcRowBytes = width * utils::safeCast<decltype(width)>(sizeof(soft_renderer::SoftRenderer::PixelType));
            for (int y = 0; y < height; ++y) {
                std::memcpy(pixels, srcData, utils::autoCast(srcRowBytes));
                pixels += sdlPitch;
                srcData += srcRowBytes;
            }
        }

        CALL_SDL(RenderClear, renderer.get());
        CALL_SDL(RenderTexture, renderer.get(), texture.get(), nullptr, nullptr);
        CALL_SDL(RenderPresent, renderer.get());
    }
    return EXIT_SUCCESS;
}
