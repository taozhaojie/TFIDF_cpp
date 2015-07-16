# TFIDF_cpp
An Implementation of TF-IDF in C++

* Need to modify the function `loadData()` to fit the real situation.
* Two versions of outputs
 - `Eigen::MatrixXf` object.
 - `std::vector<std::vector<double>>` object.


## /lyric_similarity
Use TF-IDF in music lyric similarity calculation.

* Both single thread and multi thread versions.
* Compile with g++ for single thread version
```
g++ -std=c++0x -Wall -o lyricSimilarity lyricSimilarity.cpp -static-libstdc++
```
* Compile the multithreading version
```
g++ -std=c++0x -Wall -o lyricSimilarity_multithreading lyricSimilarity_multithreading.cpp -static-libstdc++ -lpthread
```
