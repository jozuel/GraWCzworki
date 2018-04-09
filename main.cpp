/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   main.cpp
 * Author: jozuel
 *
 * Created on April 9, 2018, 8:01 PM
 */

// Gra w czwórki

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

/*
0 is player 1 is AI
*/
int main()
{
	DataWriting file;
	FourGame game;
	string fileName;
	vector<int> oneHot;
	int random, column, win = 0;	//testing actualization
	srand(time(NULL));
	while (!file.getIsFileOpen())		//open file
	{
		cout << "insert file name: ";
		getline(cin, fileName);
		file.openFile(fileName);
		if (!file.getIsFileOpen())
			cout << "cannot open file. Try insert another one." << endl;
	}
	while ((game.getCounter() < 49) && (win == 0))
	{
		do
		{
			random = rand() % 7;
			column = game.put(random);
		} while (column == -1);	//while error

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
			if(game.getPlayer()==2)
				file.convertToOneHot(3, 3, oneHot);	//win one-hot is 100 when we win 010 when no one win 001 when AI wins
			else
				file.convertToOneHot(1, 3, oneHot);
			file.writeToFile(oneHot);
		}
		if (win == 0)
		{
			file.convertToOneHot(2, 3, oneHot);
			file.writeToFile(oneHot);
		}
		game.display();
		cout << endl;
		game.setNextPlayerTurn();
	}
	file.closeFile();
	//game.put();
	getchar();
	return 0;
}

