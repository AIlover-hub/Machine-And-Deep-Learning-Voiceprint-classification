dirname = "F:/librispeech_finetuning"
import os
import sys
import threading


class MyCliear():
    # 存储找到的垃圾文件
    __FilePaths = []
    # 存储线程
    __Threads = []

    def __init__(self):
        super(MyCliear, self).__init__()
        return

    def __del__(self):
        return

    def FindFile(self, FilePath):
        # 传进文件路径 然后 找到全部的路径传入 进入File里
        for i in os.listdir(FilePath):
            bufpath = FilePath + "/" + i
            # 如果是文件夹
            if os.path.isdir(bufpath):
                self.FindFile(bufpath)
            # 如果是文件
            if os.path.isfile(bufpath):
                self.__FilePaths.append(bufpath)

    def ThFindFirst(self, FilePath):
        # 多线程 寻找
        for i in os.listdir(FilePath):
            FileName = FilePath + "/" + i
            if os.path.isdir(FileName):
                th = threading.Thread(target=self.FindFile, args=(FileName,))
                self.__Threads.append(th)
                th.start()
            elif os.path.isfile(FileName):
                self.__FilePaths.append(FileName)
        for i in self.__Threads:
            i.join()

    # 返回所以文件
    def GetPaths(self):
        return self.__FilePaths

    def GetSearchPath(self, filePath):
        for i in self.__FilePaths:
            # 截取最后的后缀名  如果后缀名在这个列表中 就迭代返回
            if ('.' + i.split(".")[-1]) in filePath:
                yield i
        return

    def DeleteFile(self, Files):

        for i in Files:
            try:
                os.remove(i)
            except Exception as e:
                raise ("移除文件可能出错，也可能没有权限", e)


if __name__ == "__main__":
    Ffun = MyCliear()
    Ffun.ThFindFirst(dirname)
    paths = Ffun.GetPaths()
    result = Ffun.GetSearchPath([".txt"])
    Ffun.DeleteFile(result)