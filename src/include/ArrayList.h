#include "scalar.h"

/**
   Class that holds a list of arrays in a two dimensional matrix.

   @author Komahan Boopathy
*/
class ArrayList{
 public:
  // Constructor and desctructor
  ArrayList(int num_tuples, int nvars);
  ~ArrayList();

  // Member functions
  void addEntry(const int *tuple);
  void getEntries(int **entries);
  int getNumEntries();

 private:
  // Member variables
  int **table;
  int max_num_tuples;
  int max_tuple_length;
  int num_entries;
};
