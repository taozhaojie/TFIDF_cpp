#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include <string>
#include <set>
#include <vector>
#include <map>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/foreach.hpp>

using namespace std;

class tfidf {

private:
	std::vector<std::string> tracks; // all songs in order
	std::vector<std::vector<double>> dataMat; // converted bag of words matrix
	unsigned int nrow; // matrix row number
	unsigned int ncol; // matrix column number
	std::vector<std::vector<double>> weightMat; //tfidf weight matrix
	std::vector<std::vector<std::string>> rawDataSet; // raw data
	std::vector<std::string> vocabList; // all terms
	std::map<std::string, int> h_hot; // hot num
	std::vector<int> numOfTerms; // used in tf calculation
	std::vector<std::string> stopWrods;

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

	std::vector<double> bagOfWords2VecMN(std::vector<std::string> & inputSet)
	{
		std::vector<double> returnVec(vocabList.size(), 0);
		for (std::string word : inputSet)
		{
			size_t idx = std::find(vocabList.begin(), vocabList.end(), word) - vocabList.begin();
			if (idx == vocabList.size())
				cout << "word: " << word << "not found" << endl;
			else
				returnVec.at(idx) += 1;
		}
		return returnVec;
	}

	void vec2mat()
	{
		cout << "Converting text to vector..." << endl;
		int cnt(0);
		for (auto it = rawDataSet.begin(); it != rawDataSet.end(); ++ it)
		{
			cnt ++;
			cout << cnt << "\r";
			std::cout.flush();
			dataMat.push_back(bagOfWords2VecMN(*it));
			numOfTerms.push_back(it->size());
			it->clear();
		}
		cout << endl;
		ncol = dataMat[0].size();
		nrow = dataMat.size();
		rawDataSet.clear(); // release memory
	}

	std::vector<std::string> textParse(std::string & bigString)
	{
		std::vector<std::string> vec;
		boost::tokenizer<> tok(bigString);
		for(boost::tokenizer<>::iterator beg = tok.begin(); beg != tok.end(); ++ beg)
		{
		    if (!(std::binary_search(stopWrods.begin(), stopWrods.end(), *beg)))
		    	vec.push_back(*beg);
		}
		return vec;
	}

	std::vector<double> vec_sum(const std::vector<double>& a, const std::vector<double>& b)
	{
	    assert(a.size() == b.size());
	    std::vector<double> result;
	    result.reserve(a.size());
	    std::transform(a.begin(), a.end(), b.begin(), 
	                   std::back_inserter(result), std::plus<double>());
	    return result;
	}

	// larger -> better
	double cosine_similarity(std::vector<double> &v1, std::vector<double> &v2)
	{
		double d1, d2, d3 = 0;
		unsigned int len = v1.size();
		for (unsigned int i = 0; i < len; ++i)
		{
			d1 += (v1[i] * v2[i]);
			d2 += (v1[i] * v1[i]);
			d3 += (v2[i] * v2[i]);
		}
		return d1 / (sqrt(d2) * sqrt(d3));
	}

	void orderByHot(std::vector<std::string> *vec)
	{
		std::map<int, std::vector<std::string>> temp;
		for (std::string e : *vec)
		{
			temp[h_hot[e]].push_back(e);
		}
		vec->clear();
		for (auto it = temp.rbegin(); it != temp.rend(); ++ it)
		{
			for (std::string e : it->second)
				vec->push_back(e);
		}
	}

public:
	unsigned int recAmount;
	unsigned int finishCount;

	void loadData()
	{
		cout << "Loading data..." << endl;
		ifstream in("track_lyrics.csv");
		string tmp;
		std::vector<std::string> vec_str;
		int cnt = 0;
		while (!in.eof()) {
			cnt ++;
			if (cnt > 500) // FOR TEST ONLY
				break;
			getline(in, tmp, '\n');
			if (tmp == "") break;
			boost::split(vec_str, tmp, boost::is_any_of(","));
			std::vector<std::string> wordList = textParse(vec_str[1]);
			rawDataSet.push_back(wordList);
			tracks.push_back(vec_str[0]);
			tmp.clear();
			vec_str.clear();
		}

		std::ifstream in2("song_hot_num.csv");
		while (!in2.eof()) 
		{
			getline(in2, tmp, '\n');
			if (tmp == "") break;
			boost::split(vec_str,tmp,boost::is_any_of(","));
			h_hot[vec_str[0]] = atoi(vec_str[1].c_str());
			tmp.clear();
			vec_str.clear();
		}
	}

	void loadStopWords()
	{
		ifstream in("stop_words.txt");
		string tmp;
		while (!in.eof()) {
			getline(in, tmp, '\n');
			stopWrods.push_back(tmp);
			tmp.clear();
		}
		std::sort(stopWrods.begin(), stopWrods.end());
	}

	void getMat()
	{
		cout << "Total " << rawDataSet.size() << " songs." << endl;
		cout << "Processing..." << endl;
		createVocabList();
		vec2mat();
		cout << "Calculating TF-IDF weight matrix..." << endl;
		std::vector<std::vector<double>> dataMat2(dataMat);
		std::vector<double> termCount;
		termCount.resize(ncol);
		
		for (unsigned int i = 0; i != nrow; ++i)
		{
			for (unsigned int j = 0; j != ncol; ++j)
			{
				if (dataMat2[i][j] > 1) // only keep 1 and 0
					dataMat2[i][j] = 1;
			}
			termCount = vec_sum(termCount, dataMat2[i]); // no. of doc. each term appears
		}
		dataMat2.clear(); //release

		std::vector<double> row_vec;
		for (unsigned int i = 0; i != nrow; ++i)
		{
			cout << "\r" << (i + 1);
			std::cout.flush();
			for (unsigned int j = 0; j != ncol; ++j)
			{
				double tf = dataMat[i][j] / numOfTerms[i];
				double idf = log((double)nrow / (termCount[j]));
				row_vec.push_back(tf * idf); // TF-IDF equation
			}
			weightMat.push_back(row_vec);
			row_vec.clear();
		}
		nrow = weightMat.size();
		cout << endl;
	}

	void saveMat(std::string filename)
	{
		cout << "Saving weight matrix to " << filename << "..." << endl;
		std::ofstream outfile;
		outfile.open(filename, std::ios_base::app);
		for (auto it = weightMat.begin(); it != weightMat.end(); ++ it)
		{
			std::ostringstream ss;
			for (auto it2 = it->begin(); it2 != it->end(); ++ it2)
			{
				ss << *it2 << ",";
			}
			outfile << ss.str().substr(0, ss.str().size()-1) << endl;
			ss.clear();
		}
	}

	void calSimi(unsigned int limit1, unsigned int limit2)
	{
		cout << "Calculating..." << endl;
		double similarity;
		std::map<double, std::vector<std::string>> simiSong;
		std::vector<std::string> recSong;
		for (std::size_t i = limit1; i != limit2; ++ i) // for each track
		{
			finishCount ++;
			for (std::size_t j = 0; j != nrow; ++ j) // compare with each track
			{
				if (j != i) // exclude track itself
				{
					similarity = cosine_similarity(weightMat[i], weightMat[j]);
					simiSong[similarity].push_back(tracks[j]);
				}
			}

			if (0 < simiSong.rbegin()->first)
			{
				for (auto it2 = simiSong.rbegin(); it2 != simiSong.rend(); ++ it2)
				{
					if (recSong.size() >= recAmount)
						break;
					std::vector<std::string> temp(it2->second);
					orderByHot(&temp);
					for (std::string e : temp)
						recSong.push_back(e);
				}
				if (recSong.size() >= recAmount)
				{
					std::vector<std::string> recSong2(recSong.begin(), recSong.begin() + recAmount);
					std::ofstream outfile;
					outfile.open("simi_songs_lyrics", std::ios_base::app);
					outfile << tracks[i] << "," << boost::join(recSong2, ",") << endl;
					outfile.close();
					recSong2.clear();
				}
				else
				{
					cout << "Unexcepted small size recommendation list." << endl;
				}	
			}
			recSong.clear();
			simiSong.clear();
		}
		cout << endl;
	}

};

int main()
{
	tfidf lyric;
	lyric.loadStopWords();
	lyric.loadData();
	lyric.recAmount = 20;
	lyric.getMat();
	lyric.saveMat("tfidf_matrix");
	lyric.calSimi(0, 10);
}