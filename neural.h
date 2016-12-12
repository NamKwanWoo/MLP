#pragma once
#ifndef _NEURALH
#define _NEURALH

#include <math.h> //sigmoid() and fabs()
#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "dataReader.h"

/*helper macros to locate the appropriate weight at the weights vector*/
#define inputToHidden(inp,hid)                weights.at(inputN*hid+inp)
#define hiddenToHidden(toLayer,fromHid,toHid)   weights.at(inputN*hiddenN + ((toLayer-2)*hiddenN*hiddenN)+hiddenN*fromHid+toHid)
#define hiddenToOutput(hid,out)               weights.at(inputN*hiddenN + (hiddenL-1)*hiddenN*hiddenN + hid*outputN+out)


/*Helper macros just as above, but for the previous Weights*/
#define _prev_inputToHidden(inp,hid)                prWeights.at(inputN*hid+inp)
#define _prev_hiddenToHidden(toLayer,fromHid,toHid)   prWeights.at(inputN*hiddenN+ ((toLayer-2)*hiddenN*hiddenN)+hiddenN*fromHid+toHid)
#define _prev_hiddenToOutput(hid,out)               prWeights.at(inputN*hiddenN + (hiddenL-1)*hiddenN*hiddenN + hid*outputN+out)


/*helper macro to locate the appropriate hidden neuron*/
#define hiddenAt(layer,hid)                     hiddenNeurons[(layer-1)*hiddenN + hid]

/*helper macros to locate the appropriate neuron's delta*/
#define outputDeltaAt(out)                      (*(odelta+out))
#define hiddenDeltaAt(layer,hid)                (*(hdelta+(layer-1)*hiddenN+hid))

/*math help*/
#define sigmoid(value)  (1/(1+exp(-value)));
#define dersigmoid(value) (value*(1-value))

class MLP
{
private:
	std::vector<float>  inputNeurons;
	std::vector<float>  hiddenNeurons;
	std::vector<float>  outputNeurons;
	std::vector<float>  weights;
	dataReader  reader;
	dataReader TT;
	int inputN, outputN, hiddenN, hiddenL;
	int target;
	double output_Sum;

public:
	MLP(int hiddenL, int hiddenN);
	~MLP();

	//assigns values to the input neurons
	bool populateInput(int fileNum);
	//calculates the whole network, from input to output
	void calculateNetwork();
	//trains the network according to our parameters
	bool trainNetwork(float teachingStep, float lmse, float momentum, int);

	//recalls the network for a given bitmap file
	void recallNetwork(int fileNum);

	void test();
};

#endif