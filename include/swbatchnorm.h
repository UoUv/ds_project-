#ifndef BN_TYPE_H_
#define BN_TYPE_H_


typedef struct BNData_st {
  void * xnorm;
  void * bottom_data;
  void * top_data;
  void * mean_by_channel;
  void * variance_by_channel;
  void * temp_mutable;
  void * bottom_diff;
  void * top_diff;
  //void * xnorm,
  float eps;
  int num;            //