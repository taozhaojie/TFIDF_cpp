#include <iostream>
#include <fstream>
#include <set>
#include <Eigen/Dense>
#include <boost/tokenizer.hpp>
#include <sstream>

using namespace std;

class tfidf {
private:
	Eigen::MatrixXf dataMat;
	int nrow; // matrix row number
	int ncol; // matrix column number
	std::vector<std::vector<std::string>> rawDataSet;
	std::vector<std::string> vocabList;

	std::string readFileText(std::string & filename)
	{
		std::ifstream in(filename);
		std::string str((std::istreambuf_iterator<char>(in)),
			            std::istreambuf_iterator<char>());
		return str;
	}

	void createVocabList()
	{
		std::set<std::string> vocabListSet;
		for (std::vector<std::string> document : rawDataSet)
		{
			for (std::string word : document)
				vocabListSet.insert(word);
		}
		std::copy(vocabListSet.begin(), vocabListSet.end(), std::back_inserter(vocabList));
	}

	Eigen::VectorXf bagOfWords2VecMN(std::vector<std::string> & inputSet)
	{
		std::vector<float> returnVec(vocabList.size(), 0);
		for (std::string word : inputSet)
		{
			size_t idx = std::find(vocabList.begin(), vocabList.end(), word) - vocabList.begin();
			if (idx == vocabList.size())
				cout << "word: " << word << "not found" << endl;
			else
				returnVec.at(idx) += 1;
		}
		Eigen::Map<Eigen::VectorXf> v(returnVec.data(),returnVec.size());
		return v;
	}

	void vec2mat()
	{
		std::vector<Eigen::VectorXf> vec;
		for (std::vector<std::string> document : rawDataSet)
		{
			vec.push_back(bagOfWords2VecMN(document));
		}
		ncol = vec[0].size();
		nrow = vec.size();
		dataMat.resize(nrow, ncol);
		for (int i = 0; i < nrow; ++i)
		{
			dataMat.row(i) = vec[i];
		}
		rawDataSet.clear(); // release memory
	}

	std::vector<std::string> textParse(std::string & bigString)
	{
		std::vector<std::string> vec;
		boost::tokenizer<> tok(bigString);
		for(boost::tokenizer<>::iterator beg = tok.begin(); beg != tok.end(); ++ beg)
		{
		    vec.push_back(*beg);
		}
		return vec;
	}

public:
	void loadData()
	{
		for (int i = 1; i != 26; ++i)
		{
			std::ostringstream ss;
			ss << "test/" << i << ".txt";
			std::string filename = ss.str();
			std::string str = readFileText(filename);
			std::vector<std::string> wordList = textParse(str);
			rawDataSet.push_back(wordList);
		}
	}

	Eigen::MatrixXf getMat()
	{
		createVocabList();
		vec2mat();
		
		Eigen::MatrixXf dataMat2(dataMat);
		Eigen::VectorXf termCount;
		termCount.resize(ncol);
		for (int i = 0; i != nrow; ++i)
		{
			for (int j = 0; j != ncol; ++j)
			{
				if (dataMat2(i,j) > 1) // only keep 1 and 0
					dataMat2(i,j) = 1;
			}
			termCount += dataMat2.row(i); // no. of doc. each term appears
		}
		dataMat2.resize(0,0); //release

		Eigen::MatrixXf returnMat; // a new matrix have same size with dataMat
		returnMat.resize(nrow, ncol);
		for (int i = 0; i != nrow; ++i)
		{
			for (int j = 0; j != ncol; ++j)
			{
				double tf = dataMat(i,j) / (dataMat.row(i).sum());
				double idf = log((double)nrow / (termCount(j)));
				returnMat(i,j) = tf * idf;
			}
		}
		return returnMat;
	}

};

int main()
{
	tfidf ins;
	ins.loadData();
	Eigen::MatrixXf mat = ins.getMat();
}
