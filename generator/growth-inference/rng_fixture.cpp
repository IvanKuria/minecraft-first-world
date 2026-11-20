#define main growth_main
#include "growth.cpp"
#undef main
int main(){RNG r(190513);int bounds[]={256,64,256,6,3,2147483647};for(int i=0;i<10000;i++)std::cout<<r.nextInt(bounds[i%6])<<"\n";}
