// Gra w czwórki
//#define saveMod
#include "stdafx.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <time.h>
#include "FourGame.h"
#include "DataWriting.h"
#include <sstream>
#include <vector>

using namespace std;

void displayVector(vector<int> vec)
{
	for (int i : vec)
	{
		cout << i << ",";
	}
}
/*
0 is player 1 is AI
*/
int main()
{
	DataWriting file;
	FourGame game;
	string fileName;
	vector<int> oneHot;
	int random, column, win = 0, input;	//testing actualisation
	srand(time(NULL));
#ifdef seveMod
	while (!file.getIsFileOpen())		//open file
	{
		cout << "insert file name: ";
		getline(cin, fileName);
		file.openFile(fileName);
		if (!file.getIsFileOpen())
			cout << "cannot open file. Try insert anotherone." << endl;
	}
	for (int i = 0; i < 100000; i++)
	{
		win = 0;
		while ((game.getCounter() < 49) && (win == 0))
		{
			do
			{
				random = rand() % 7;
				column = game.put(random);
			} while (column == -1);	//while error
			oneHot.clear();
			file.convertToOneHot(random + 1, 7, oneHot);	//save row and column to file
			file.writeToFile(oneHot);
			oneHot.clear();
			file.convertToOneHot(column + 1, 7, oneHot);
			file.writeToFile(oneHot);
			oneHot.clear();

			//players turn
			file.convertToOneHot(game.getPlayer(), 2, oneHot);
			file.writeToFile(oneHot);
			oneHot.clear();
			if ((game.checkDiagonal() != 0) || (game.checkHorizontalAndVertical() != 0))
			{
				win = 1;
				if (game.getPlayer() == 2)
					file.convertToOneHot(3, 3, oneHot);	//win one-hot is 100 when we win 010 when noone win 001 when AI wins
				else
					file.convertToOneHot(1, 3, oneHot);
				file.writeToFile(oneHot);
			}
			oneHot.clear();
			if (win == 0)
			{
				file.convertToOneHot(2, 3, oneHot);
				file.writeToFile(oneHot);
			}
			//game.display();
			//cout << endl;
			game.setNextPlayerTurn();
		}
		file.writeNewLineToFile();
		oneHot.clear();
		game.resetGame();
		if (i % 1000 == 0)
		{
			cout << i << "of 100000" << endl;
		}
	}
	file.closeFile();
#endif // seveMod
	while ((game.getCounter() < 49) && (win == 0))
	{
		do
		{
			cin >> input;
			column = game.put(input);
		} while (column == -1);	//while error
		displayVector(oneHot);
		oneHot.clear();
		file.convertToOneHot(input + 1, 7, oneHot);	//display row and column to send input to python
		displayVector(oneHot);
		oneHot.clear();
		file.convertToOneHot(column + 1, 7, oneHot);
		displayVector(oneHot);
		oneHot.clear();

			//players turn
		file.convertToOneHot(game.getPlayer(), 2, oneHot);
		displayVector(oneHot);
		oneHot.clear();
		if ((game.checkDiagonal() != 0) || (game.checkHorizontalAndVertical() != 0))
		{
			win = 1;
			if (game.getPlayer() == 2)
				file.convertToOneHot(3, 3, oneHot);	//win one-hot is 100 when we win 010 when noone win 001 when AI wins
			else
				file.convertToOneHot(1, 3, oneHot);
			displayVector(oneHot);

			cout <<endl<<"player "<< game.getPlayer() << " wins";
		}
		oneHot.clear();
		if (win == 0)
		{
			file.convertToOneHot(2, 3, oneHot);
			displayVector(oneHot);
		}
		cout << endl;
		game.display();
		cout << endl;
		game.setNextPlayerTurn();
	}
	//game.put();
	getchar();
	cin.ignore();
	return 0;
}

