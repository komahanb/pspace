#include<stdio.h>
#include"ArrayList.h"

/**
   Constructor implementation for ArrayList

   @param max_num_tuples
   @param max_tuple_length
*/
ArrayList::ArrayList( int max_num_tuples, int max_tuple_length ){
  this->max_num_tuples  = max_num_tuples;
  this->max_tuple_length  = max_tuple_length;
  this->table = new int*[max_num_tuples];
  for (int i = 0 ; i < max_num_tuples; i++){
    this->table[i] = new int[max_tuple_length];
  }
  this->num_entries = 0;
}

/**
   Destructor for ArrayList
*/
ArrayList::~ArrayList(){
  for (int i = 0 ; i < max_num_tuples; i++){
    delete [] this->table[i];
  }
  delete [] this->table;
}

/**
   Returns the number of entries in the list
*/
int ArrayList::getNumEntries(){
  return num_entries;
}

/**
   Adds the tuple in to the list
   @param tuple
*/
void ArrayList::addEntry(const int *tuple){
  for (int i = 0; i < this->max_tuple_length; i++){
    this->table[this->num_entries][i] = tuple[i];
  }
  this->num_entries++;
}

/**
   Gets the pointer to list entries
*/
void ArrayList::getEntries(int **entries){
  // Copy values for return
  for (int j = 0; j < this->num_entries; j++){
    for (int i = 0; i < this->max_tuple_length; i++){
      entries[j][i] = this->table[j][i];
    }
  }
}

/*
  Test function
*/
void mainz( int argc, char *argv[] ){

  ArrayList *dlist = new ArrayList(5,2);

  int d1[] = {2,4};
  dlist->addEntry(d1);
  int d2[] = {5,9};

  dlist->addEntry(d2);
  dlist->addEntry(d1);
  printf("get entries %d\n", dlist->getNumEntries());


  // Allocate return array
  int **indx = new int*[5];
  for (int i = 0 ; i < 5; i++){
    indx[i] = new int[2];
  }

  dlist->getEntries(indx);
  if(indx){
    printf("allocated\n");
  }

  printf("%d %d\n", indx[0][0], indx[0][1]);
  printf("%d %d\n", indx[1][0], indx[1][1]);

  for (int i = 0 ; i < 5; i++){
    delete [] indx[i];
  }

  delete [] indx;
  delete dlist;

}
