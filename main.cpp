#include <iostream>
#include <ctime>
#include "neural.h"
#include "dataReader.h"
using namespace std;

int main()
{
	int hiddenNeurons, hiddenLayers, trainingFiles;
	float teachingStep, leastMSE, momentum;
	
	// 5 4 0.0001 0.65

	cout << "Input HiddenNeurons and HiddenLayers\n" << endl;
	cout << "Hidden Neurons:\t";
	cin >> hiddenNeurons;
	cout << "Hidden Layers:\t";
	cin >> hiddenLayers;

	cout << endl;

	cout << "Input Learning Rate and Momentum\n" << endl;
	cout << "Learning Rate:\t";
	cin >> teachingStep;
	cout << "Momentum:\t";
	cin >> momentum;
	cout << endl;

	leastMSE = 0.34;
	trainingFiles = 60290;

	MLP* neuralnet = new MLP(hiddenLayers, hiddenNeurons);

	if (!(neuralnet->trainNetwork(teachingStep, leastMSE, momentum, trainingFiles)))
	{
	}

	int number = 0;

	/*
	while (number <= trainingFiles)
	{
		if(number % 1000 == 0)
			neuralnet->recallNetwork(number);
		number++;
	}*/

	neuralnet->test();

	getchar();
	return 0;
}


