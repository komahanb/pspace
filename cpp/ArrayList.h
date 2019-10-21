class ArrayList{
 public:
  ArrayList(int num_tuples, int nvars);
  ~ArrayList();
  void addEntry(const int *tuple);
  void getEntries(int **entries);
  int getNumEntries();
  
 private:
  int **table;
  int max_num_tuples;
  int max_tuple_length;
  int num_entries;
};
