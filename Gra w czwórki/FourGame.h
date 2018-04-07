#pragma once
class FourGame
{
private:
	int tab[7][7];
	int player;
	int counter;
public:
	FourGame();
	~FourGame();
	int getCounter();
	int getPlayer();
	void display();
	void empty();
	int put(int column);
	int checkHorizontalAndVertical();
	int checkDiagonal();
	void setNextPlayerTurn();
};

