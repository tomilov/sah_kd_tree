#include <builder/builder.hpp>
#include <compute/compute.hpp>
#include <compute/make.hpp>
#include <scene_data/scene_data.hpp>
#include <scene_loader/scene_loader.hpp>
#include <soft_renderer/soft_renderer.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
#include <utils/scope_guard.hpp>

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <fmt/base.h>
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

builder::TreePtr makeTree(QString sceneFileName)
{
    compute::CudaDevicePtr cudaDevice = compute::makeCudaDevice(std::nullopt);
    scene_data::SceneData sceneData;
    QFileInfo sceneFileInfo{sceneFileName};
    if ((false)) {
        if (!scene_loader::load(sceneData, sceneFileInfo)) {
            qCDebug(softRendererMain).noquote() << u"Cannot load scene from file %1"_s.arg(sceneFileName);
            return nullptr;
        }
    } else {
        auto cachePath = QStandardPaths::writableLocation(QStandardPaths::CacheLocation);
        if (!scene_loader::cachingLoad(sceneData, sceneFileInfo, cachePath.isEmpty() ? QDir::temp() : cachePath)) {
            qCDebug(softRendererMain).noquote() << u"Cannot load scene from file %1"_s.arg(sceneFileName);
            return nullptr;
        }
    }
    builder::Tree::Settings settings = {
        .emptinessFactor = kEmptinessFactor,
        .traversalCost = kTraversalCost,
        .intersectionCost = kIntersectionCost,
        .maxTreeDepth = kMaxTreeDepth,
    };
    const auto progress = [start = std::chrono::steady_clock::now()](size_t progressValue)
    {
        using namespace std::chrono_literals;
        if (start + 10s < std::chrono::steady_clock::now()) {
            INVARIANT(false, "{}", progressValue);
        }
        return false;
    };
    builder::Tree tree{settings, *cudaDevice, std::make_shared<scene_data::SceneData>(std::move(sceneData)), progress};
    if (tree.isEmpty()) {
        return nullptr;
    }
    return builder::makeTreePtr(std::move(tree));
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
std::unique_ptr<std::FILE, decltype(&std::fclose)> openFile [[maybe_unused]] (const char * filepath, std::FILE * stream)
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

}  // namespace

int main(int argc, char * argv[])
{
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        SPDLOG_ERROR("SDL_Init failed: {}", SDL_GetError());
        return EXIT_FAILURE;
    }
    utils::ScopeGuard sdlQuit{SDL_Quit};

    gli::extent2d::value_type width = 1024;
    gli::extent2d::value_type height = 768;
    constexpr SDL_WindowFlags kWindowFlags = SDL_WINDOW_RESIZABLE;
    std::unique_ptr<SDL_Window, decltype(&SDL_DestroyWindow)> window{SDL_CreateWindow(APPLICATION_NAME, utils::autoCast(width), utils::autoCast(height), kWindowFlags), &SDL_DestroyWindow};
    if (!window) {
        SPDLOG_ERROR("SDL_CreateWindow failed: {}", SDL_GetError());
        return EXIT_FAILURE;
    }
    // SDL_SetWindowRelativeMouseMode(window.get(), true);
    if (!SDL_SetWindowMinimumSize(window.get(), 16, 16)) {
        SPDLOG_ERROR("SDL_SetWindowMinimumSize failed: {}", SDL_GetError());
        return EXIT_FAILURE;
    }

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
    soft_renderer::SoftRenderer softRenderer{"default"sv, kClearColor};
    {
        INVARIANT(argc > 1, "{}", argc);
        builder::TreePtr tree = makeTree(QString::fromUtf8(argv[1]));
        if (!tree) {
            return EXIT_FAILURE;
        }
        softRenderer.setTree(std::move(*tree));
    }

    constexpr glm::float32 kMoveSpeed = 3.0f;
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
        // glm::quat orientation = glm::conjugate(glm::toQuat(glm::lookAt(position, glm::vec3{position.x, position.y, 0.0f}, glm::vec3{0.0f, -1.0f, 0.0f})));
        frameSettings.position = {yaw, pitch, 0.0f};
        glm::vec3 eulerAngles{0.0f, 0.0f, 0.0f};
        frameSettings.orientation = glm::quat{glm::radians(eulerAngles)};
    }

    const auto createTarget = [&width, &height]() -> gli::texture2d
    {
        return {soft_renderer::SoftRenderer::kTargetFormat, gli::extent2d{width, height}};
    };
    auto target = createTarget();

    Uint64 timePrev = SDL_GetPerformanceCounter();
    const Uint64 freq = SDL_GetPerformanceFrequency();

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
                target = createTarget();
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
                    const bool isCtrlPressed = (event.key.mod & SDL_KMOD_CTRL) != 0;
                    keyS = isPressed && !isCtrlPressed;
                    if (isCtrlPressed && !event.key.repeat) {
                        if (argc > 2) {
                            auto rgbTarget = gli::convert(target, gli::format::FORMAT_RGB8_UNORM_PACK8);
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
                        }
                    }
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
                default: {
                    break;
                }
                }
                break;
            }
            case SDL_EVENT_MOUSE_MOTION: {
                if ((event.motion.state & SDL_BUTTON_LMASK) != 0) {
                    yaw += event.motion.xrel * kMouseSensetivity;
                    pitch += event.motion.yrel * kMouseSensetivity;
                    constexpr auto kHalfPi = glm::half_pi<glm::float32>() - 0.01f;
                    pitch = glm::clamp(pitch, -kHalfPi, kHalfPi);
                }
                break;
            }
            default: {
                break;
            }
            }
        }
        const Uint64 timeNow = SDL_GetPerformanceCounter();
        const glm::float32 dt = static_cast<glm::float32>(timeNow - timePrev) / static_cast<glm::float32>(freq);
        timePrev = timeNow;

        const glm::quat qYaw = glm::angleAxis(yaw, glm::vec3{0.0f, 1.0f, 0.0f});
        const glm::quat qPitch = glm::angleAxis(pitch, glm::vec3{1.0f, 0.0f, 0.0f});
        frameSettings.orientation = glm::normalize(qYaw * qPitch);

        const glm::vec3 forward = frameSettings.orientation * glm::vec3{0.0f, 0.0f, 1.0f};
        const glm::vec3 right = frameSettings.orientation * glm::vec3{1.0f, 0.0f, 0.0f};
        const glm::vec3 up{0.0f, 1.0f, 0.0f};

        glm::vec3 moveDir{0.0f};
        if (keyW) moveDir += forward;
        if (keyS) moveDir -= forward;
        if (keyD) moveDir += right;
        if (keyA) moveDir -= right;
        if (keyE) moveDir += up;
        if (keyQ) moveDir -= up;

        if (glm::dot(moveDir, moveDir) > 0.0f) {
            moveDir = glm::normalize(moveDir);
        }
        frameSettings.position += moveDir * kMoveSpeed * dt;

        softRenderer.render(frameSettings, target);

        uint8_t * pixels = nullptr;
        int sdlPitch = 0;
        if (SDL_LockTexture(texture.get(), nullptr, utils::autoCast(&pixels), &sdlPitch)) {
            utils::ScopeGuard sdlUnlockTexture{SDL_UnlockTexture, texture.get()};
            const uint8_t * srcData = utils::autoCast(target.data(0, 0, 0));
            const auto srcRowBytes = width * utils::safeCast<decltype(width)>(sizeof(soft_renderer::SoftRenderer::PixelType));
            for (int y = 0; y < height; ++y) {
                std::memcpy(pixels, srcData, utils::autoCast(srcRowBytes));
                pixels += sdlPitch;
                srcData += srcRowBytes;
            }
        }

        if (!SDL_RenderClear(renderer.get())) {
            SPDLOG_WARN("SDL_RenderClear failed: {}", SDL_GetError());
        }
        if (!SDL_RenderTexture(renderer.get(), texture.get(), nullptr, nullptr)) {
            SPDLOG_WARN("SDL_RenderTexture failed: {}", SDL_GetError());
        }
        if (!SDL_RenderPresent(renderer.get())) {
            SPDLOG_WARN("SDL_RenderPresent failed: {}", SDL_GetError());
        }
    }
    return EXIT_SUCCESS;
}
