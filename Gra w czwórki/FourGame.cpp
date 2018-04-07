#include "stdafx.h"
#include "FourGame.h"
#include <cstdio>
#include <iostream>
using namespace std;
int FourGame::getCounter()
{
	return counter;
}
int FourGame::getPlayer()
{
	return player;
}
void FourGame::display()
{
	for (int i = 0; i < 7; i++)
	{
		for (int j = 0; j < 7; j++)
		{
			cout << tab[i][j] << " ";
		}
		cout << endl;
	}
}
FourGame::FourGame()
{
	player = 1;
	empty();
}
/*
set all elements to 0
*/
void FourGame::empty()
{
	for (int i = 0; i < 7; i++)
	{
		for (int j = 0; j < 7; j++)
		{
			tab[i][j] = 0;
		}
	}
}
/*
put player number in selected column. If max(7) is reached return -1 else return row iterator(From 0). It works like stack
*/
int FourGame::put(int column)
{
	int i = 0;
	if (tab[6][column] != 0)
	{
		return -1;
	}
	else
	{
		while ((tab[i][column] != 0) && (i < 7))
		{
			i++;
		}
		tab[i][column] = player;
	}
	counter++;
	return i;
}
/*
check if there are 4 points in the row. Horizontal or Vertical. If is then return 1 else return 0
*/
int FourGame::checkHorizontalAndVertical()
{
	int horizontalCounter = 0;
	int verticalCounter = 0;
	for (int i = 0; i < 7; i++)
	{
		horizontalCounter = 0;
		verticalCounter = 0;
		for (int j = 0; j < 7; j++)
		{
			if (tab[i][j] == player)
			{
				horizontalCounter++;
			}
			else
			{
				horizontalCounter = 0;
			}
			if (horizontalCounter == 4)
			{
				return 1;
			}
			if (tab[j][i] == player)
			{
				verticalCounter++;
			}
			else
			{
				verticalCounter = 0;
			}
			if (verticalCounter == 4)
			{
				return 1;
			}
		}
	}
	return 0;
}
int FourGame::checkDiagonal()
{
	int counter = 0;
	for (int j = -6; j < 7; j++)
	{
		counter = 0;			//it should by from top to down and then from lower to upper
		for (int i = 0; i + j < 7; i++)	//from lower to upper
		{
			if (i + j >= 0) 
			{
				if (tab[i][i + j] == player)
				{
					counter++;
				}
				else
				{
					counter = 0;
				}
				if (counter == 4)
				{
					return 1;
				}
			}
		}
	}
	for (int j = 12; j > 0; j--)	//expand column iterator so i iterator can reach corners
	{
		counter = 0;		//it should by from down to top and then from upper to lower
		for (int i = 0; j - i >= 0; i++)	//from upper to lower
		{
			if (j - i < 7) 
			{
				if (tab[i][j - i] == player)
				{
					counter++;
				}
				else
				{
					counter = 0;
				}
				if (counter == 4)
				{
					return 1;
				}
			}
		}
	}
	return 0;
}
/*
sets next players turn. Players are 1 and 2
*/
void FourGame::setNextPlayerTurn()
{
	player = (player) % 2  + 1;
}
FourGame::~FourGame()
{
}
