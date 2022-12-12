import os
import sys
import shutil
import traceback

class Renamer:
    inputPath = ""
    outputPath = ""
    colsRowsOrder = True
    inputGrid = [0,0]

    def printHelpAndExit(self):
        print("This script renames arbitrary input lightfield files into the right format for the encoding.")
        print("Run as: python renameLFImages inputPath outputPath inputOrder gridWidth gridHeight")
        print("inputPath and outputPath - paths to the existing directories")
        print("inputOrder - the order of the images in the grid - 0 is row major and 1 column major")
        print("gridWidth and gridHeight - number of images in the grid in X and Y axis")
        exit(0)

    def checkArgs(self):
        ARGS_NUM = 5
        if len(sys.argv) < ARGS_NUM+1  or sys.argv[1] == "-h" or sys.argv[1] == "--help":
            self.printHelpAndExit()
        self.inputPath = sys.argv[1]
        self.outputPath = sys.argv[2]
        self.colsRowsOrder = bool(sys.argv[3])
        self.inputGrid[0] = int(sys.argv[4])
        self.inputGrid[1] = int(sys.argv[5])

    def getName(self, imgID):
       col = imgID % self.inputGrid[0]
       row = imgID // self.inputGrid[0]
       if self.colsRowsOrder:
            return str(col)+"_"+str(row)
       else:
            return str(row)+"_"+str(col)

    def convert(self):
        files = sorted(os.listdir(self.inputPath))
        imgID = 0
        for file in files:
           inFile = self.inputPath
           inFile = os.path.join(inFile, file)
           extension = os.path.splitext(file)[1]
           name = self.getName(imgID)+extension
           outFile = self.outputPath
           outFile = os.path.join(outFile, name)
           shutil.copyfile(inFile, outFile)
           imgID += 1

    def run(self):
        self.checkArgs()
        self.convert()

try:
    renamer = Renamer()
    renamer.run()
except Exception as e:
    print(e)
    print(traceback.format_exc())

