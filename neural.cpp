#include "Neural.h"
#include "dataReader.h"
#include <ctime>
using namespace std;

MLP::MLP(int hL, int hN)
{
	outputN = 1; // Binary Classification 0 or 1
	hiddenL = hL;
	hiddenN = hN + 1;
	
	//seed random number generator
	//initialize the filereader
	reader.loadDataFile("trn.txt", 13, 1);		// Feature 13, Target 1

	//read the first image to see what kind of input will our net have
	inputN = 13;

	//let's allocate the memory for the weights
	weights.reserve(inputN*hiddenN + (hiddenN*hiddenN*(hiddenL - 1)) + hiddenN*outputN);

	//also let's set the size for the neurons vector
	inputNeurons.resize(inputN);
	hiddenNeurons.resize(hiddenN*hiddenL);
	outputNeurons.resize(outputN);
	srand((unsigned int)time(0));

	//randomize weights for inputs to 1st hidden layer
	for (int i = 0; i < inputN*hiddenN; i++)
	{
		weights.push_back(((rand() % 200) - 100) / 10000.0);//[-0.5,0.5]
	}
	//if there are more than 1 hidden layers, randomize their weights
	for (int i = 1; i < hiddenL; i++)
	{
		for (int j = 0; j < hiddenN*hiddenN; j++)
		{
			weights.push_back(((rand() % 200) - 100) / 10000.0);//[-0.5,0.5]
		}
	}
	//and finally randomize the weights for the output layer
	for (int i = 0; i < hiddenN*outputN; i++)
	{
		weights.push_back(((rand() % 200) - 100) / 10000.0);//[-0.5,0.5]
	}
	
	//for (auto i = weights.begin(); i != weights.end(); ++i)
		//cout << *i << endl;
	cout << endl;
}

MLP::~MLP()
{
	weights.clear();
	inputNeurons.clear();
	hiddenNeurons.clear();
	outputNeurons.clear();
}

void MLP::calculateNetwork()
{
	//let's propagate towards the hidden layer
	for (int hidden = 0; hidden < hiddenN; hidden++)
	{
		hiddenAt(1, hidden) = 0;		// 1st hidden Neuron
		for (int input = 0; input < inputN; input++)
			hiddenAt(1, hidden) += inputNeurons.at(input)*inputToHidden(input, hidden); //Weigthed Sum
		hiddenAt(1, hidden) = sigmoid(hiddenAt(1, hidden));
	}

	//now if we got more than one hidden layers
	for (int i = 2; i <= hiddenL; i++)
	{
		//for each one of these extra layers calculate their values
		for (int j = 0; j < hiddenN; j++)//to
		{
			hiddenAt(i, j) = 0;		//Initialize
			for (int k = 0; k <hiddenN; k++)//from
				hiddenAt(i, j) += hiddenAt(i - 1, k)*hiddenToHidden(i, k, j);						//Weigthed Sum
			hiddenAt(i, j) = sigmoid(hiddenAt(i, j));
		}
	}

	//and now hidden to output
	for (int i = 0; i< outputN; i++)
	{
		outputNeurons.at(i) = 0;
		for (int j = 0; j <hiddenN; j++)
			outputNeurons.at(i) += hiddenAt(hiddenL, j) * hiddenToOutput(j, i);				//Weigthed Sum
		output_Sum = outputNeurons.at(i);
		outputNeurons.at(i) = sigmoid(outputNeurons.at(i));
	}
}

//assigns values to the input neurons
bool MLP::populateInput(int fileNum)
{
	dataEntry *data = reader.data.at(fileNum);		//get data Vector

	for (int i = 0; i < inputN; i++)						//inputN = 13
		inputNeurons.at(i) = (float)(data->pattern[i]);
	target = (data->target[0]);						//Target is 0 or 1
	return true;
}

//trains the network according to our parameters
bool MLP::trainNetwork(float teachingStep, float lmse, float momentum, int trainingFiles)
{
	float mse = 999.0;			//Mean Square Error
	int tCounter = 0;			//test Counter
	int epochs = 1;				//Epoch
	float error = 0.0;			//Error
	int tp, tn;

	float* odelta = (float*)malloc(sizeof(float)*outputN);		//output delta
	float* hdelta = (float*)malloc(sizeof(float)*hiddenN*hiddenL);		//Hidden delta

	//a buffer for the weights
	std::vector<float> tempWeights = weights;
	//used to keep the previous weights before modification, for momentum
	std::vector<float> prWeights = weights;
	
	while ( epochs < 200 )//fabs(mse - lmse) > 0.1)
	{
		mse = 0.0;
		tp = 0;
		tn = 0;

		//for each file  Total Output
		while (tCounter < trainingFiles)
		{
			//first populate the input neurons
			if (!populateInput(tCounter))
			{
				printf("An error has been encountered while reading a waveform file \n\r");
				return false;
			}

			//then calculate the network
			calculateNetwork();

/*=============================Feedforward Propagation=============================*/

			//Now we have calculated the network for this iteration
			//let's back-propagate following the back-propagation algorithm
			
			/*
			for (int i = 0; i < outputN; i++)
			{
				//let's get the delta of the output layer
				//and the accumulated error
				
				if (i == 0)//target) 
				{
					outputDeltaAt(i) = (0.0 - outputNeurons[i])*dersigmoid(outputNeurons[i]);
					error += (0.0 - outputNeurons[i])*(0.0 - outputNeurons[i]);
				}
				else
				{
					outputDeltaAt(i) = (1.0 - outputNeurons[i])*dersigmoid(outputNeurons[i]);
					error += (1.0 - outputNeurons[i])*(1.0 - outputNeurons[i]);
				}
			}*/

			outputDeltaAt(0) = (target - outputNeurons[0]) * dersigmoid(outputNeurons[0]);
			error += (target - outputNeurons[0]) * (target - outputNeurons[0])/2;

			double prediction = outputNeurons[0];

			if (prediction > 0.5)
			{
				if (target == 1)
					tp++;
			}
			else if (prediction < 0.5)
			{
				if (target == 0)
					tn++;
			}
/*=============================Feedforward End==================================*/


			//we start popagating backwards now, to get the error of each neuron
			//in every layer


/*=============================BackPropagation=============================*/
			
			//let's get the delta of the last hidden layer first
			for (int i = 0; i < hiddenN; i++)
			{
				hiddenDeltaAt(hiddenL, i) = 0;//zero the values from the previous iteration
				 //add to the delta for each connection with an output neuron
				for (int j = 0; j < outputN; j++)
					hiddenDeltaAt(hiddenL, i) += outputDeltaAt(j) * hiddenToOutput(i, j);
	
				//The derivative here is only because of the
				//delta rule weight adjustment about to follow
				hiddenDeltaAt(hiddenL, i) *= dersigmoid(hiddenAt(hiddenL, i));
			}

			//now for each additional hidden layer, provided they exist
			for (int i = hiddenL - 1; i >0; i--)
			{
				//add to each neuron's hidden delta
				for (int j = 0; j < hiddenN; j++)//from
				{
					hiddenDeltaAt(i, j) = 0;//zero the values from the previous iteration

					for (int k = 0; k <hiddenN; k++)//to
					{
						//the previous hidden layers delta multiplied by the weights
						//for each neuron
						hiddenDeltaAt(i, j) += hiddenDeltaAt(i + 1, k) * hiddenToHidden(i + 1, j, k);
					}

					//The derivative here is only because of the
					//delta rule weight adjustment about to follow
					hiddenDeltaAt(i, j) *= dersigmoid(hiddenAt(i, j));
				}
			}

			//Weights modification
			tempWeights = weights;//keep the previous weights somewhere, we will need them

			//hidden to Input weights
			for (int i = 0; i < inputN; i++)
				for (int j = 0; j < hiddenN; j++)
					inputToHidden(i, j) += momentum*(inputToHidden(i, j) - _prev_inputToHidden(i, j)) + teachingStep* hiddenDeltaAt(1, j) * inputNeurons[i];


			//hidden to hidden weights, provided more than 1 layer exists
			for (int i = 2; i <= hiddenL; i++)
				for (int j = 0; j < hiddenN; j++)					//from
					for (int k = 0; k < hiddenN; k++)				//to
						hiddenToHidden(i, j, k) += momentum*(hiddenToHidden(i, j, k) - _prev_hiddenToHidden(i, j, k)) + teachingStep * hiddenDeltaAt(i, k) * hiddenAt(i - 1, j);

			//last hidden layer to output weights
			for (int i = 0; i < outputN; i++)
				for (int j = 0; j < hiddenN; j++)
					hiddenToOutput(j, i) += momentum*(hiddenToOutput(j, i) - _prev_hiddenToOutput(j, i)) + teachingStep * outputDeltaAt(i) * hiddenAt(hiddenL, j);

			prWeights = tempWeights;

/*=============================End BackPropagation=============================*/
			if (tCounter % 30000 == 0)
			{
				cout << outputNeurons[0] << "\t" << target << endl;
			}
/*=============================Get MSE, ACCURACY=============================*/

			//add to the total mse for this epoch
			mse += (error/60290.0);  /// (outputN + 1);
			//zero out the error for the next iteration
			error = 0;
			tCounter++;
		}// Epoch 1 End


		//reset the counter
		tCounter = 0;
		double accuracy = ((tp + tn) / 60290.0) * 100;

		if (epochs % 5 == 0)
		{
			printf("Epoch:  %d\tMean square error:  %.7lf \n", epochs, pow(mse, 0.5));
			printf("Accuracy:  %lf\n\n", accuracy);
		}
	
		epochs++;
	}
	cout << "End Training \nIf you want to test, please input keyboard...." << endl;
	getchar();
	return true;
}
/*============================= End of Training ==============================*/

void MLP::recallNetwork(int fileNum)
{
	//first populate the input neurons
	populateInput(fileNum);

	//then calculate the network
	calculateNetwork();

	float winner = -1;
	int index = 0;

	//find the best fitting output
	for (int i = 0; i < outputN; i++)
		if (outputNeurons[i] > winner)
		{
			winner = outputNeurons[i];
			index = i;
		}

	//output it
	//printf("The neural network thinks that image %d represents a \n\r\n\r \t\t----->| %d |<------\t\t \n\r\n\r", fileNum, index);

	//now let's the exact percentages of what it thnks
	printf(" %d we see  %d%% probability |\n\\", fileNum, (int)(outputNeurons[0] * 100));
}

void MLP::test()
{
	TT.loadDataFile("tst.txt", 13, 1);		// Feature 13, Target 1
	int tCounter = 0;
	int tp=0, tn=0, fp=0, fn=0;

	//for each file  Total Output
	while (tCounter < 18000)
	{
		dataEntry *data = TT.data.at(tCounter);		//get data Vector

		for (int i = 0; i < inputN; i++)						//inputN = 13
			inputNeurons.at(i) = (float)(data->pattern[i]);
		target = (data->target[0]);						//Target is 0 or 1

		//then calculate the network
		calculateNetwork();
		
		cout << outputNeurons[0]  << "\t\t" <<  target << endl;

		if (outputNeurons[0] > 0.5)
		{
			if (target == 1)
				tp++;
			else
				fp++;
		}
		else if (outputNeurons[0] < 0.5)
		{
			if (target == 0)
				tn++;
			else
				fn++;
		}

		tCounter++;
	}
	
	double accuracy = (double)(tp + tn) / (tp + tn + fp + fn);
	double recall = (double)(tp) / (tp + fp);
	double precision = (double)(tp) / (tp + fn);

	cout.precision(6);
	printf("TP: %d\tTN: %d\tFP: %d\t\tFN: %d\n\n", tp, tn, fp, fn);
	printf("Accuracy: %lf\tRecall: %lf\tPrecision: %lf\n\n", accuracy*100.0, recall*100.0, precision*100.0);
	getchar();
}