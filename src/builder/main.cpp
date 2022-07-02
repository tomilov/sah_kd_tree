#include <builder/builder.hpp>

#include <QtCore>

#include <cstdlib>

int main(int argc, char * argv[])
{
    QCoreApplication application{argc, argv};

    QCommandLineParser commandLineParser;

    commandLineParser.addPositionalArgument("scene", "Scene file name.", "file");

    QCommandLineOption useCacheOption("use-cache", "Use cache.");
    commandLineParser.addOption(useCacheOption);

    QCommandLineOption cachePathOption("cache-path", "Cache path.", "cachePath", "");
    commandLineParser.addOption(cachePathOption);

    QCommandLineOption emptinessFactorOption("emptiness-factor", "SAH factor to encourage algorithm to cut off empty space.", "emptinessFactor", "0");
    commandLineParser.addOption(emptinessFactorOption);

    QCommandLineOption traversalCostOption("traversal-cost", "SAH traversal step cost.", "traversalCost", "0");
    commandLineParser.addOption(traversalCostOption);

    QCommandLineOption intersectionCostOption("intersection-cost", "SAH intersection cost.", "intersectionCost", "0");
    commandLineParser.addOption(intersectionCostOption);

    QCommandLineOption maxDepthOption("max-depth", "Max kd-tree depth.", "maxDepth", "0");
    commandLineParser.addOption(maxDepthOption);

    commandLineParser.process(application);

    const auto buildTree = [&] {
        QString cachePath = commandLineParser.value(cachePathOption);
        const QStringList args = commandLineParser.positionalArguments();
        if (args.size() < 1) {
            return QCoreApplication::exit(2);
        }
        bool ok = false;
        float emptinessFactor = commandLineParser.value(emptinessFactorOption).toFloat(&ok);
        if (!ok) {
            return QCoreApplication::exit(3);
        }
        float traversalCost = commandLineParser.value(traversalCostOption).toFloat(&ok);
        if (!ok) {
            return QCoreApplication::exit(4);
        }
        float intersectionCost = commandLineParser.value(intersectionCostOption).toFloat(&ok);
        if (!ok) {
            return QCoreApplication::exit(5);
        }
        int maxDepth = commandLineParser.value(maxDepthOption).toInt(&ok);
        if (!ok) {
            return QCoreApplication::exit(6);
        }
        bool useCache = commandLineParser.isSet(useCacheOption);
        if (!useCache && cachePath.isEmpty()) {
            if (!builder::buildSceneFromFile(args.at(0), emptinessFactor, traversalCost, intersectionCost, maxDepth)) {
                return QCoreApplication::exit(EXIT_FAILURE);
            }
        } else {
            if (!builder::buildSceneFromFileOrCache(args.at(0), cachePath, emptinessFactor, traversalCost, intersectionCost, maxDepth)) {
                return QCoreApplication::exit(EXIT_FAILURE);
            }
        }
        return QCoreApplication::quit();
    };
    QTimer::singleShot(0, qApp, buildTree);
    return application.exec();
}
