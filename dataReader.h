#pragma once
#ifndef _DATAREADER
#define _DATAREADER

#include <vector>
#include <string>

/*******************************************************************
* stores a data item
********************************************************************/
class dataEntry
{
public:

	double *pattern;    //input patterns
	int *target;        //target result

public:
	dataEntry(double *p, int *t) : pattern(p), target(t) {}

	~dataEntry()
	{
		delete[] pattern;
		delete[] target;
	}

};

/*******************************************************************
* Training Sets Storage - stores shortcuts to data items
********************************************************************/
class trainingDataSet
{
public:
	std::vector<dataEntry *> trainingSet;
	std::vector<dataEntry *> generalizationSet;
	std::vector<dataEntry *> validationSet;

	trainingDataSet() {}

	void clear()
	{
		trainingSet.clear();
		generalizationSet.clear();
		validationSet.clear();
	}
};

//dataset retrieval approach enum
enum
{
	NONE, STATIC, GROWING, WINDOWING
};

//data reader class
class dataReader
{

	//private members
	//----------------------------------------------------------------------------------------------------------------
public:

	//data storage
	std::vector<dataEntry *> data;
	int nInputs;
	int nTargets;

	//current data set
	trainingDataSet tSet;

	//data set creation approach and total number of dataSets
	int creationApproach;
	int numTrainingSets;
	int trainingDataEndIndex;

	//creation approach variables
	double growingStepSize;            //step size - percentage of total set
	int growingLastDataIndex;        //last index added to current dataSet
	int windowingSetSize;            //initial size of set
	int windowingStepSize;            //how many entries to move window by
	int windowingStartIndex;        //window start index	

public:

	dataReader() : creationApproach(NONE), numTrainingSets(-1) {}

	~dataReader();

	bool loadDataFile(const char *filename, int nI, int nT);

	void setCreationApproach(int approach, double param1 = -1, double param2 = -1);

	int getNumTrainingSets();

	trainingDataSet *getTrainingDataSet();

	std::vector<dataEntry *> &getAllDataEntries();

private:

	void createStaticDataSet();

	void createGrowingDataSet();

	void createWindowingDataSet();

	void processLine(std::string &line);
};

#endif
