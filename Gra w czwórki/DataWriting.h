#pragma once
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>
class DataWriting
{
private:
	std::ofstream file;
	bool isFileOpen;
	std::vector<int> vec;
public:
	DataWriting();
	DataWriting(std::string name);
	~DataWriting();
	std::vector<int> getVec();
	void openFile(std::string name);
	bool getIsFileOpen();
	void writeToFile(std::string text);
	void writeNewLineToFile();
	void convertToOneHot(int number, int max,std::vector<int>& vec);
	void closeFile();
	void writeToFile(std::vector<int> vec);
};

