#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
using namespace std;
static constexpr size_t HS=1u<<16, WINDOW=65535, MINM=4, MAXM=130; static size_t DEPTH=48;
static constexpr int32_t NONE=-1;
static inline size_t h3(const vector<uint8_t>& b,size_t p){uint32_t x=b[p],y=b[p+1],z=b[p+2];return ((((x*0x1e35a7bdU)^(y<<11)^(z<<3))>>16)&(HS-1));}
static inline size_t offbytes(size_t d){return d<128?1:(d<16384?2:3);}
vector<uint8_t> enc(const vector<uint8_t>& in){
 vector<uint8_t> out; out.reserve(in.size()+9); out.insert(out.end(),{'P','Y','L','Z',2}); uint32_t n=in.size(); for(int i=0;i<4;i++)out.push_back((n>>(8*i))&255);
 vector<int32_t> head(HS,NONE),prev(in.size(),NONE); vector<uint8_t> lits; lits.reserve(128); size_t p=0;
 auto flush=[&](){size_t q=0;while(q<lits.size()){size_t c=min<size_t>(128,lits.size()-q);out.push_back((uint8_t)(c-1));out.insert(out.end(),lits.begin()+q,lits.begin()+q+c);q+=c;}lits.clear();};
 auto find=[&](size_t at)->pair<size_t,size_t>{
  if(at+MINM>in.size())return {0,0};size_t bl=0,bo=0;size_t h=h3(in,at);int32_t c=head[h];size_t probes=0;
  while(c!=NONE&&probes<DEPTH){size_t ci=(size_t)c,d=at-ci;if(d>WINDOW)break;size_t l=0;while(l<MAXM&&at+l<in.size()&&in[ci+l]==in[at+l])l++;
   if(l>=MINM&&l>bl){bl=l;bo=d;if(l==MAXM)break;}c=prev[ci];probes++;}return {bl,bo};};
 auto insert=[&](size_t at){if(at+2>=in.size())return;size_t h=h3(in,at);prev[at]=head[h];head[h]=(int32_t)at;};
 while(p<in.size()){
  auto [bl,bo]=find(p);
  if(bl>=MINM&&bl<16&&p+1<in.size()){auto [nl,no]=find(p+1);if(nl>=MINM){size_t lc=lits.empty()?2:1;size_t cg=bl>1+offbytes(bo)?bl-(1+offbytes(bo)):0;size_t ng=nl>lc+1+offbytes(no)?nl-(lc+1+offbytes(no)):0;if(ng>cg)bl=0;}}
  if(bl>=MINM){flush();out.push_back((uint8_t)(128+bl-3));size_t d=bo;while(d>=128){out.push_back((uint8_t)((d&127)|128));d>>=7;}out.push_back((uint8_t)d);size_t end=p+bl;while(p<end){insert(p);p++;}}
  else{lits.push_back(in[p]);insert(p);p++;if(lits.size()==128)flush();}
 }
 flush();return out;
}
vector<uint8_t> dec(const vector<uint8_t>& x){
 if(x.size()<9||!equal(x.begin(),x.begin()+4,"PYLZ")||(x[4]!=1&&x[4]!=2))throw 1;
 uint8_t ver=x[4];uint32_t n=0;for(int i=0;i<4;i++)n|=(uint32_t)x[5+i]<<(8*i);
 vector<uint8_t> o;o.reserve(n);size_t p=9;
 while(o.size()<n){
  if(p>=x.size())throw 2;uint8_t t=x[p++];
  if(t<128){size_t l=t+1;if(p+l>x.size()||o.size()+l>n)throw 3;o.insert(o.end(),x.begin()+p,x.begin()+p+l);p+=l;}
  else{
   size_t l=(t&127)+3;if(l<4)throw 4;size_t d=0;
   if(ver==1){if(p+2>x.size())throw 4;d=x[p]|(x[p+1]<<8);p+=2;}
   else{unsigned shift=0;bool done=false;for(int k=0;k<3;k++){if(p>=x.size())throw 4;uint8_t b=x[p++];d|=(size_t)(b&127)<<shift;if(!(b&128)){if(k>0&&d<(size_t(1)<<(7*k)))throw 4;done=true;break;}shift+=7;}if(!done)throw 4;}
   if(!d||d>65535||d>o.size()||o.size()+l>n)throw 5;for(size_t j=0;j<l;j++)o.push_back(o[o.size()-d]);
  }
 }
 if(p!=x.size())throw 6;return o;
}
int main(int argc,char**argv){ifstream f(argv[1],ios::binary);vector<uint8_t> in((istreambuf_iterator<char>(f)),{}); auto t=chrono::steady_clock::now(); size_t chosen=32, best=SIZE_MAX; if(in.size()>=4096){vector<uint8_t> sample(in.begin(),in.begin()+min<size_t>(65536,in.size())); for(size_t d: {32u,48u,64u}){DEPTH=d; auto trial=enc(sample); if(trial.size()<best){best=trial.size();chosen=d;}}} DEPTH=chosen; auto z=enc(in);auto t1=chrono::steady_clock::now();auto out=dec(z);auto t2=chrono::steady_clock::now();if(out!=in)return 7;if(argc>2){ofstream g(argv[2],ios::binary);g.write((const char*)z.data(),z.size());}double e=chrono::duration<double>(t1-t).count(),d=chrono::duration<double>(t2-t1).count();cout<<"profile="<<chosen<<" input="<<in.size()<<" encoded="<<z.size()<<" ratio="<<(double)z.size()/in.size()<<" encode_s="<<e<<" encode_MiB_s="<<(in.size()/1048576.0/e)<<" decode_s="<<d<<" decode_MiB_s="<<(in.size()/1048576.0/d)<<" roundtrip=OK\n";}
