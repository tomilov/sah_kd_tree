#include "builder.hpp"

#include <QtCore>

#include <cstdlib>

int main(int argc, char * argv[])
{
    QCoreApplication application{argc, argv};

    QCommandLineParser commandLineParser;

    commandLineParser.addPositionalArgument("scene", "Scene file name.");

    QCommandLineOption useCacheOption("use-cache", "Use cache.");
    commandLineParser.addOption(useCacheOption);

    commandLineParser.process(application);

    const QStringList args = commandLineParser.positionalArguments();

    const auto buildTree = [&] {
        if (args.size() < 1) {
            return QCoreApplication::exit(2);
        }
        if (build(args.at(0), commandLineParser.isSet(useCacheOption))) {
            return QCoreApplication::quit();
        } else {
            return QCoreApplication::exit(EXIT_FAILURE);
        }
    };
    QTimer::singleShot(0, qApp, buildTree);
    return application.exec();
}
