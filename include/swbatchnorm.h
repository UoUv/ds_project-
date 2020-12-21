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
  int num;            //batch_size
  int channels;      //C
  int spatial_dim;     //H*W

}BNData;

void sw_batch_norm_use_forward_impl_f(
    float * bottom_data,
    float * top_data,
    float * blobs_0,
    float * blobs_1,
    float * blobs_2,
    float * mean_by_channel,
    float * variance_by_channel,
    float * temp_mutable,
    float * xnorm,
    float eps,
    int num,            //batch_size
    int channels,       //C
    int spatial_dim     //H*W
);

void sw_batch_norm_nouse_forward_impl_f(
    float * bottom_data,
    float * top_data,
    float * num_by_chans_,
    float * temp_mutable,
    float * blobs_0,
    float * blobs_1,
    float * blobs_2,
    float * moving_average_fraction_,
    float * xnorm,
    float eps,
    int num,
    int channels,
    int spatial_dim
);

void sw_batch_norm_use_backward_impl_f(
    int temp_count,
    int num,
    int channels,
    int spatial_dim,
    const float * top_diff,
    const float * temp_cpu_data,
    float * bottom_diff
);

void sw_batch_norm_nouse_backward_impl_f(
    const float * top_data,
    const float * top_diff,
    float * bottom_diff,
    int num,
    int channels,
    int spatial_dim,
    int temp_count,
    const float * temp_cpu_data
);


void sw_batch_norm_use_forward_impl_d(
    