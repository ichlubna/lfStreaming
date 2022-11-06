#include "decoder.h"
#include "libs/arguments/arguments.hpp"

int main(int argc, char **argv)
{
    Arguments args(argc, argv);
    std::string path = static_cast<std::string>(args["-i"]);

    std::string helpText{ "Usage:\n"
                          "Example: lfEncoder -i /MyAmazingMachine/thoseImages -q 1.0 -f H265 -o ./coolFile.lf\n"
                          "-i - encoded LF file\n"
                          "Use mouse to change the viewing angle.\n"
                        };
    if(args.printHelpIfPresent(helpText))
        return 0;

    if(path == "")
    {
        std::cerr << "No input specified. Use -h for help." << std::endl;
        return EXIT_FAILURE;
    }

    try
    {
        Decoder decoder(path);
        decoder.decodeAndPlay();
    }
    catch(const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;

}
