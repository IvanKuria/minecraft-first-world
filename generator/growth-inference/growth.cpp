// Hypothesis family only. Not recovered Minecraft source.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
struct RNG {uint64_t state,calls=0;explicit RNG(uint64_t seed):state((seed^0x5DEECE66DULL)&((1ULL<<48)-1)){} uint32_t bits(int n){state=(state*0x5DEECE66DULL+11)&((1ULL<<48)-1);calls++;return state>>(48-n);} int nextInt(int n){if((n&-n)==n)return (int)((n*(uint64_t)bits(31))>>31);uint32_t b,v;do{b=bits(31);v=b%n;}while(b-v+(n-1)>0x7fffffff);return v;}};
int main(int argc,char**argv){if(argc<8){std::cerr<<"mode mono6|mono26|voter6 boundary free|fixed seeds updates seed output width(optional)\n";return 2;}std::string mode=argv[1],boundary=argv[2],out=argv[7];int seeds=std::stoi(argv[3]);uint64_t updates=std::stoull(argv[4]),seed=std::stoull(argv[5]); // argv6 label, reserved description
int W=argc>8?std::stoi(argv[8]):256,D=W,H=W/4,N=W*D*H;RNG r(seed);std::vector<uint8_t>a(N,0);auto idx=[&](int x,int y,int z){return (y*D+z)*W+x;};auto wall=[&](int x,int y,int z){return x==0||x==W-1||z==0||z==D-1||y==0;};
for(int i=0;i<seeds;i++){int x=r.nextInt(W),y=r.nextInt(H),z=r.nextInt(D);a[idx(x,y,z)]=1;}
if(boundary=="fixed")for(int y=0;y<H;y++)for(int z=0;z<D;z++)for(int x=0;x<W;x++)if(wall(x,y,z))a[idx(x,y,z)]=1;
uint64_t initcalls=r.calls,initstate=r.state;size_t initialsolid=std::count(a.begin(),a.end(),1);auto t0=std::chrono::steady_clock::now();std::array<int,6>dx={1,-1,0,0,0,0},dy={0,0,1,-1,0,0},dz={0,0,0,0,1,-1};
for(uint64_t i=0;i<updates;i++){int x=r.nextInt(W),y=r.nextInt(H),z=r.nextInt(D),xx=x,yy=y,zz=z;if(mode=="mono26"){xx+=r.nextInt(3)-1;yy+=r.nextInt(3)-1;zz+=r.nextInt(3)-1;}else{int dir=r.nextInt(6);xx+=dx[dir];yy+=dy[dir];zz+=dz[dir];}if(xx<0||xx>=W||yy<0||yy>=H||zz<0||zz>=D||(boundary=="fixed"&&wall(x,y,z)))continue;uint8_t src=a[idx(xx,yy,zz)];if(mode=="voter6"||src)a[idx(x,y,z)]=src;}
size_t solid=0,single=0,top_islands=0,wallsolid=0,walltotal=0,interfaces=0;std::vector<int>layers(H);for(int y=0;y<H;y++)for(int z=0;z<D;z++)for(int x=0;x<W;x++){int v=a[idx(x,y,z)];solid+=v;layers[y]+=v;if(wall(x,y,z)){wallsolid+=v;walltotal++;}if(v){int neighbors=0;for(int d=0;d<6;d++){int xx=x+dx[d],yy=y+dy[d],zz=z+dz[d];if(xx>=0&&xx<W&&yy>=0&&yy<H&&zz>=0&&zz<D)neighbors+=a[idx(xx,yy,zz)];}if(!neighbors)single++;if(y==H-1&&x>0&&x<W-1&&z>0&&z<D-1&&!a[idx(x-1,y,z)]&&!a[idx(x+1,y,z)]&&!a[idx(x,y,z-1)]&&!a[idx(x,y,z+1)])top_islands++;}if(x+1<W)interfaces+=v!=a[idx(x+1,y,z)];if(y+1<H)interfaces+=v!=a[idx(x,y+1,z)];if(z+1<D)interfaces+=v!=a[idx(x,y,z+1)];}
std::ofstream f(out,std::ios::binary);f.write((char*)a.data(),N);f.close();double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();std::cout<<"{\"mode\":\""<<mode<<"\",\"boundary\":\""<<boundary<<"\",\"width\":"<<W<<",\"height\":"<<H<<",\"seed_experimental\":"<<seed<<",\"seed_placements\":"<<seeds<<",\"initial_solid\":"<<initialsolid<<",\"updates\":"<<updates<<",\"initial_rng_calls\":"<<initcalls<<",\"total_rng_calls\":"<<r.calls<<",\"state_after_initialization\":"<<initstate<<",\"state_after_generation\":"<<r.state<<",\"solid\":"<<solid<<",\"face_interfaces\":"<<interfaces<<",\"isolated_solid_voxels\":"<<single<<",\"isolated_top_cells_4neighbors\":"<<top_islands<<",\"boundary_solid\":"<<wallsolid<<",\"boundary_total\":"<<walltotal<<",\"seconds\":"<<secs<<",\"layers\":[";for(int y=0;y<H;y++){if(y)std::cout<<",";std::cout<<layers[y];}std::cout<<"]}\n";
}
