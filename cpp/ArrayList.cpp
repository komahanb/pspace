#include<stdio.h>
#include"ArrayList.h"

ArrayList::ArrayList(int max_num_tuples, int max_tuple_length){
  this->max_num_tuples  = max_num_tuples;
  this->max_tuple_length  = max_tuple_length;  
  this->table = new int*[max_num_tuples];
  for (int i = 0 ; i < max_num_tuples; i++){
    this->table[i] = new int[max_tuple_length];
  }
  this->num_entries = 0;
}

ArrayList::~ArrayList(){
  for (int i = 0 ; i < max_num_tuples; i++){
    delete [] this->table[i];
  }
  delete [] this->table;
}

int ArrayList::getNumEntries(){
  return num_entries;
}

void ArrayList::addEntry(const int *tuple){
  for (int i = 0; i < this->max_tuple_length; i++){
    this->table[this->num_entries][i] = tuple[i];
  }
  this->num_entries++;
}

void ArrayList::getEntries(int **entries){
  // Copy values for return
  for (int j = 0; j < this->num_entries; j++){
    for (int i = 0; i < this->max_tuple_length; i++){
      entries[j][i] = this->table[j][i];
      //       printf("%d ", this->table[j][i]);
    }
    // printf("\n");
  }
}

/*

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

  // if(*indx[0]){
    
  // };
  
  //  for (int j = 0; j < dlist->getNumEntries(); j++){
  // for (int i = 0; i < 2; i++){
  printf("%d %d\n", indx[0][0], indx[0][1]);
  printf("%d %d\n", indx[1][0], indx[1][1]);
  
  // }     
      //}
  
  delete dlist;
}
