#include "stdafx.h"
#include "DataWriting.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sstream>

using namespace std;

DataWriting::DataWriting()
{
	isFileOpen = false;
}
DataWriting::DataWriting(string name)
{
	isFileOpen = false;
	openFile(name);
}
vector<int> DataWriting::getVec()
{
	return vec;
}
/*
gets isFileOpen value. Returns true if file is open.
*/
bool DataWriting::getIsFileOpen()
{
	return isFileOpen;
}
/*
writes text to file
*/
void DataWriting::writeToFile(string text)
{
	if(file.is_open())
		file << text.c_str();
}

/*
if file is opened then close it and open new one. Then set isFileOpen variable
*/
void DataWriting::openFile(string name)
{
	closeFile();
	file.open(name + ".txt");
	file.is_open() ? isFileOpen = true : isFileOpen = false;
}
/*
converts number to it's one-hot equivalent. For example number = 3 and max = 5 will be 001000
*/
void DataWriting::convertToOneHot(int number, int max,vector<int>& oneHot)
{
	//vector<int> oneHot;
	for (int i = 1; i < number; i++)	//sets 0 to all position up to number
	{
		oneHot.push_back(0);
	}
	oneHot.push_back(1);				//sets number at correct position
	for (int i = number + 1; i <= max; i++) //sets at other position 0
	{
		oneHot.push_back(0);
	}
}
void DataWriting::closeFile()
{
	if (file.is_open())
		file.close();
}
void DataWriting::writeNewLineToFile()
{
	file << endl;
}
void DataWriting::writeToFile(vector<int> vec)
{
	string s;
	stringstream stream;
	int num;
	char bufer[50];
	if (isFileOpen)
	{
		for (int i = 0; i < vec.size(); i++)
		{
			num = vec[i];
			stream << num<<",";
			s += std::to_string((long long)num); //+ ",";
			file << to_string(num);
			//file << ',';
		}
	}
	/*for(int &i : vec)
	{
		s = to_string(i) + ',';
		//s = "0,";
		file << s;
		//file.put(s.c_str());
		//file.put(',');
		//file << s << ',';
	}*/
}
DataWriting::~DataWriting()
{
	closeFile();
}
