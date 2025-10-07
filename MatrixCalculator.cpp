#include<iostream>
#include<cstdio>
#include<cstring>
#include<utility>
#include<vector>
#include<string>
#include<cmath>
#include<algorithm>

long long read(char& ch){
    long long ans=0;
    bool flag=0;
    ch=getchar();
    if(ch=='-'){
        ch=getchar();
        flag=1;
    }
    while(1){
        if(ch<'0'||ch>'9'){
            break;
        }
        else{
            ans=(ans<<3)+(ans<<1);
            ans+=(ch-'0');
        }
        ch=getchar();
    }
    if(flag){
        return -ans;
    }
    else{
        return ans;
    }
}
long long gcd(long long a,long long b){
    if(a<b){
        std::swap(a,b);
    }
    long long c;
    while(b){
        c=a%b;
        a=b;
        b=c;
    }
    return a;
}
inline long long lcm(long long a,long long b){
    return a*b/gcd(a,b);
}
class frac{
    public:
    long long x,y;
    frac(){
        x=0ll;
        y=1ll;
    }
    frac(long long _x,long long _y):x(_x),y(_y){}
    frac(const frac& f){
        x=f.x;
        y=f.y;
    }
    frac(int _x){
        x=static_cast<long long>(_x);
        y=1ll;
    }
    void reduce(){
        if(!x){
            y=1;
            return;
        }
        long long g=gcd(abs(x),y);
        x/=g;
        y/=g;
    }
    void init(long long a,long long b){
        if(b<0){
            a=-a;
            b=-b;
        }
        x=a;
        y=b;
        reduce();
    }
    void readFrac(){
        long long a,b;
        char ch;
        a=read(ch);
        if(ch=='/'){
            b=read(ch);
            init(a,b);
        }
        else{
            init(a,1ll);
        }
    }
    void putFrac(){
        if(y==1ll){
            printf("%lld",x);
        }
        else{
            printf("%lld/%lld",x,y);
        }
    }
    frac Abs(){
        frac f;
        f.init(abs(x),y);
        return f;
    }
    frac& operator=(const int& oth){
        x=static_cast<long long>(oth);
        y=1ll;
        return *this;
    }
    frac operator+(const frac& oth)const{
        long long a=x,b=y,c=oth.x,d=oth.y;
        long long k=lcm(b,d);
        a*=(k/b);
        c*=(k/d);
        frac ans;
        ans.init(a+c,k);
        ans.reduce();
        return ans;
    }
    frac operator-(const frac& oth)const{
        long long a=x,b=y,c=oth.x,d=oth.y;
        long long k=lcm(b,d);
        a*=(k/b);
        c*=(k/d);
        frac ans;
        ans.init(a-c,k);
        ans.reduce();
        return ans;
    }
    frac operator*(const frac& oth)const{
        long long a=x,b=y,c=oth.x,d=oth.y;
        frac ans;
        ans.init(a*c,b*d);
        ans.reduce();
        return ans;
    }
    frac operator/(const frac& oth)const{
        long long a=x,b=y,c=oth.x,d=oth.y;
        frac ans;
        ans.init(a*d,b*c);
        if(ans.y<0){
            ans.y=-ans.y;
            ans.x=-ans.x;
        }
        ans.reduce();
        return ans;
    }
    bool operator>(const frac& oth)const{
        long double a,b,c,d,e,f;
        a=x;b=y;c=oth.x;d=oth.y;
        e=a/b;f=c/d;
        return e>f;
    }
    bool operator<(const frac& oth)const{
        long double a,b,c,d,e,f;
        a=x;b=y;c=oth.x;d=oth.y;
        e=a/b;f=c/d;
        return e<f;
    }
    bool operator==(const frac& oth)const{
        return x==oth.x&&y==oth.y;
    }
    bool operator==(const double& oth)const{
        double df=(double)x/(double)y;
        return df==oth;
    }
    bool operator==(const int& oth)const{
        double df=(double)x/(double)y,doth=(double)oth;
        return df==doth;
    }
    bool operator>=(const frac& oth)const{
        long long a,b,c,d,e,f;
        a=x;b=y;c=oth.x;d=oth.y;
        e=a/b;f=c/d;
        return e>f||(*this)==oth;
    }
    bool operator<=(const frac& oth)const{
        long long a,b,c,d,e,f;
        a=x;b=y;c=oth.x;d=oth.y;
        e=a/b;f=c/d;
        return e<f||(*this)==oth;
    }
    bool operator!=(const frac& oth)const{
        return !((*this)==oth);
    }
    inline double real()const{
        return double(x)/double(y);
    }
    friend std::istream& operator>>(std::istream& istr,frac& f){
        std::string s;
        int _x=0,_y=0;
        bool flag=0,sgn=0;
        istr>>s;
        int len=s.size();
        for(int i=0;i<len;i++){
            if(s[i]=='-'){
                sgn=1;
            }
            else if(s[i]=='/'){
                flag=1;
            }
            else{
                if(flag){
                    _y=_y*10+(s[i]-'0');
                }
                else{
                    _x=_x*10+(s[i]-'0');
                }
            }
        }
        if(flag){
            f.x=_x;
            f.y=_y;
        }
        else{
            f.x=_x;
            f.y=1ll;
        }
        if(sgn){
            f.x=-_x;
        }
        f.reduce();
        return istr;
    }
    friend std::ostream& operator<<(std::ostream& ostr,frac& f){
        ostr<<f.x;
        if(f.y!=1ll){
            ostr<<'/'<<f.y;
        }
        return ostr;
    }
};
template<typename T>
T Abs(T x){
    if(x<0){
        return x*(-1);
    }
    else{
        return x;
    }
}
template<typename T>
T scalarProduct(const std::vector<T>& a,const std::vector<T>& b){
    T sum=0;
    int n=std::min(a.size(),b.size());
    for(int i=0;i<n;i++){
        sum=sum+(a[i]*b[i]);
    }
    return sum;
}
template<typename T>
std::vector<T> operator+(const std::vector<T>& lhs,const std::vector<T>& rhs){
    int n=std::min(lhs.size(),rhs.size());
    std::vector<T> ret(n,0);
    for(int i=0;i<n;i++){
        ret[i]=lhs[i]+rhs[i];
    }
    return ret;
}
template<typename T>
std::vector<T> operator-(const std::vector<T>& lhs,const std::vector<T>& rhs){
    int n=std::min(lhs.size(),rhs.size());
    std::vector<T> ret(n,0);
    for(int i=0;i<n;i++){
        ret[i]=lhs[i]-rhs[i];
    }
    return ret;
}
template<typename T>
std::vector<T> operator*(T lhs,const std::vector<T>& rhs){
    int n=rhs.size();
    std::vector<T> ret(n,0);
    for(int i=0;i<n;i++){
        ret[i]=lhs*rhs[i];
    }
    return ret;
}
template<typename T>
T operator*(const std::vector<T>& lhs,const std::vector<T>& rhs){
    T sum=0;
    int n=std::min(lhs.size(),rhs.size());
    for(int i=0;i<n;i++){
        sum=sum+(lhs[i]*rhs[i]);
    }
    return sum;
}
template<typename T>
double norm(const std::vector<T>& v){
    return sqrt(double(v*v));
}
template<>
double norm(const std::vector<frac>& v){
    frac f=v*v;
    return sqrt(f.real());
}
template<typename T>
std::vector<double> unitize(const std::vector<T>& v){
    int n=v.size();
    double len=norm(v);
    std::vector<double> ret(n,0);
    for(int i=0;i<n;i++){
        ret[i]=double(v[i])/len;
    }
    return ret;
}
template<>
std::vector<double> unitize(const std::vector<frac>& v){
    int n=v.size();
    double len=norm(v);
    std::vector<double> ret(n,0);
    for(int i=0;i<n;i++){
        ret[i]=v[i].real()/len;
    }
    return ret;
}
template<typename T>
class matrix{
    public:
    int n,m;
    std::vector<std::vector<T> > mat;
    inline void init(int x,int y){
        n=x;m=y;
        std::vector<T> temp;
        T f=0;
        temp.assign(m,f);
        mat.assign(n,temp);
    }
    matrix():n(0),m(0){}
    matrix(int _n,int _m){
        init(_n,_m);
    }
    matrix(int _n){
        init(_n,_n);
    }
    void copy(const matrix& oth){
        init(oth.n,oth.m);
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                mat[i][j]=oth.mat[i][j];
            }
        }
    }
    matrix(const matrix& oth){
        copy(oth);
    }
    void copyRight(const matrix& oth){
        if(n!=oth.n){
            return;
        }
        m+=oth.m;
        for(int i=0;i<n;i++){
            for(int j=0;j<oth.m;j++){
                mat[i].push_back(oth.mat[i][j]);
            }
        }
    }
    bool isSameSize(const matrix& oth){
        if(n!=oth.n||m!=oth.m){
            return 0;
        }
        else{
            return 1;
        }
    }
    void createSquare(int x){
        init(x,x);
    }
    void createIdentity(int x){
        init(x,x);
        for(int i=0;i<x;i++){
            T f=1;
            mat[i][i]=f;
        }
    }
    void readMatrix(int n,int m){
        init(n,m);
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                std::cin>>mat[i][j];
            }
        }
    }
    void putMatrix(){
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                std::cout<<mat[i][j];
                if(j<m-1){
                    std::cout<<' ';
                }
            }
            std::cout<<'\n';
        }
    }
    std::vector<T> row(int r){
        return mat[r];
    }
    std::vector<T> col(int c){
        std::vector<T> v(n,0);
        for(int i=0;i<n;i++){
            v[i]=mat[i][c];
        }
        return v;
    }
    matrix& operator=(const matrix& oth){
        n=oth.n;
        m=oth.m;
        mat=oth.mat;
        return *this;
    }
    matrix operator*(const T& k)const{
        matrix A;
        A.copy(*this);
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                A.mat[i][j]=A.mat[i][j]*k;
            }
        }
        return A;
    }
    matrix operator+(const matrix& oth)const{
        matrix A;
        A.copy(*this);
        if(!A.isSameSize(oth)){
            return A;
        }
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                A.mat[i][j]=A.mat[i][j]+oth.mat[i][j];
            }
        }
        return A;
    }
    matrix operator-(const matrix& oth)const{
        matrix A;
        A.copy(*this);
        if(!A.isSameSize(oth)){
            return A;
        }
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                A.mat[i][j]=A.mat[i][j]-oth.mat[i][j];
            }
        }
        return A;
    }
    matrix operator*(const matrix& oth)const{
        matrix A;
        A.init(n,oth.m);
        if(m!=oth.n){
            return *this;
        }
        for(int i=0;i<n;i++){
            for(int j=0;j<oth.m;j++){
                T f=0;
                for(int k=0;k<m;k++){
                    f=f+mat[i][k]*oth.mat[k][j];
                }
                A.mat[i][j]=f;
            }
        }
        return A;
    }
    matrix quickPower(int k){
        matrix A,B;
        A.createIdentity(n);
        B.copy(*this);
        if(n!=m){
            return A;
        }
        while(k){
            if(k&1){
                A=A*B;
            }
            B=B*B;
            k>>=1;
        }
        return A;
    }
    inline matrix getRemainder(int x,int y){
        matrix A;
        A.init(n-1,m-1);
        int p=0,q=0;
        for(int i=0;i<n;i++){
            if(i==x){
                continue;
            }
            for(int j=0;j<m;j++){
                if(j==y){
                    continue;
                }
                A.mat[p][q]=mat[i][j];
                q++;
            }
            q=0;
            p++;
        }
        return A;
    }
    void gaussElimination(){
        for(int i=0;i<n;i++){
            int piv=i;
            for(int j=i;j<n;j++){
                if(Abs(mat[j][i])>Abs(mat[piv][i])){
                    piv=j;
                }
            }
            if(mat[piv][i]==0){
                return;
            }
            swap(mat[i],mat[piv]);
            for(int j=i+1;j<m;j++){
                mat[i][j]=mat[i][j]/mat[i][i];
            }
            if(mat[i][i]==0){
                return;
            }
            T f=1;
            mat[i][i]=f;
            for(int j=0;j<n;j++){
                if(i==j){
                    continue;
                }
                for(int k=i+1;k<m;k++){
                    mat[j][k]=mat[j][k]-mat[i][k]*mat[j][i];
                }
                mat[j][i]=mat[j][i]-mat[i][i]*mat[j][i];
            }
        }
    }
    T det(){
        int sgn=1;
        matrix b;
        b.copy(*this);
        for(int i=0;i<n;i++){
            if(b.mat[i][i].x==0){
                bool flag=0;
                for(int j=i+1;j<n;j++){
                    if(b.mat[i][j].x){
                        swap(b.mat[i],b.mat[j]);
                        sgn=-sgn;
                        flag=1;
                        break;
                    }
                }
                if(!flag){
                    return 0;
                }
            }
            for(int j=i+1;j<n;j++){
                T f=b.mat[j][i]/b.mat[i][i];
                for(int k=i;k<n;k++){
                    b.mat[j][k]=b.mat[j][k]-b.mat[i][k]*f;
                }
            }
        }
        T f=sgn;
        for(int i=0;i<n;i++){
            f=f*b.mat[i][i];
        }
        return f;
    }
    matrix inverse(){
        matrix r;
        r.createIdentity(n);
        matrix b;
        b.copy(*this);
        b.copyRight(r);
        b.gaussElimination();
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                r.mat[i][j]=b.mat[i][j+n];
            }
        }
        return r;
    }
    int checker(){
        //now abandoned
        for(int i=0;i<n;i++){
            bool flag=1;
            for(int j=0;j<m-1;j++){
                if(mat[i][j]==0){
                    continue;
                }
                flag=0;
                break;
            }
            if(flag){
                if(mat[i][m-1]==0){
                    return 1;
                }
                else{
                    return -1;
                }
            }
        }
        return 0;
    }
    int getVerdict(){
        int s=0,r=0;
        for(int i=0;i<n;i++){
            bool flag=0;
            for(int j=0;j<m-1;j++){
                if(mat[i][j]!=0){
                    flag=1;
                }
            }
            if(flag){
                s++;
                r++;
            }
            else if(mat[i][m-1]!=0){
                r++;
            }
        }
        if(s==r&&r==m-1){
            return 0;
        }
        else if(s==r&&r<m-1){
            return r;
        }
        else{
            return -1;
        }
    }
    matrix getSolveSet(int r){
        matrix<T> ret(m-1,m-r);
        for(int i=r;i<m-1;i++){
            for(int j=0;j<r;j++){
                ret.mat[j][i-r]=mat[j][i]*(-1);
            }
            ret.mat[i][i-r]=1;
        }
        for(int i=0;i<std::min(m-1,n);i++){
            ret.mat[i][m-r-1]=mat[i][m-1];
        }
        return ret;
    }
    void putSolveSet(){
        int cnt=m-1;
        std::cout<<"X="<<std::endl;
        for(int i=0;i<cnt;i++){
            if(i!=0){
                std::cout<<"+";
            }
            std::cout<<"k"<<i+1<<"(";
            for(int j=0;j<n;j++){
                if(j<n-1){
                    std::cout<<mat[j][i]<<",";
                }
                else{
                    std::cout<<mat[j][i];
                }
            }
            std::cout<<")^T"<<std::endl;
        }
        std::cout<<"+(";
        for(int i=0;i<n;i++){
            if(i<n-1){
                std::cout<<mat[i][cnt]<<",";
            }
            else{
                std::cout<<mat[i][cnt];
            }
        }
        std::cout<<")^T"<<std::endl;
    }
    void orthogonalize(){
        std::vector<std::vector<T> > cols(m,std::vector<T>(0));
        for(int i=0;i<m;i++){
            cols[i]=col(i);
        }
        for(int i=1;i<m;i++){
            std::vector<T> v=cols[i];
            for(int j=0;j<i;j++){
                T k=(cols[j]*cols[i])/(cols[j]*cols[j]);
                v=v-k*cols[j];
            }
            cols[i]=v;
        }
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                mat[i][j]=cols[j][i];
            }
        }
    }
    matrix<double> unitize(){
        //contains orthogonalization
        std::vector<std::vector<T> > cols(m,std::vector<T>(0));
        for(int i=0;i<m;i++){
            cols[i]=col(i);
        }
        for(int i=1;i<m;i++){
            std::vector<T> v=cols[i];
            for(int j=0;j<i;j++){
                T k=(cols[j]*cols[i])/(cols[j]*cols[j]);
                v=v-k*cols[j];
            }
            cols[i]=v;
        }
        std::vector<std::vector<double> > rcols(m,std::vector<double>(0));
        for(int i=0;i<m;i++){
            rcols[i]=::unitize(cols[i]);
        }
        matrix<double> rmat(n,m);
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                rmat.mat[i][j]=rcols[j][i];
            }
        }
        return rmat;
    }
    friend std::istream& operator>>(std::istream& istr,matrix<T>& f){
        int _n,_m;
        istr>>_n>>_m;
        f.n=_n;
        f.m=_m;
        f.mat.assign(f.n,std::vector<T>(f.m,0));
        f.n=_n;
        f.m=_m;
        for(int i=0;i<f.n;i++){
            for(int j=0;j<f.m;j++){
                istr>>f.mat[i][j];
            }
        }
        return istr;
    }
    friend std::ostream& operator<<(std::ostream& ostr,matrix<T> f){
        for(int i=0;i<f.n;i++){
            for(int j=0;j<f.m;j++){
                ostr<<f.mat[i][j];
                if(j<f.m-1){
                    ostr<<' ';
                }
            }
            ostr<<'\n';
        }
        return ostr;
    }
};
std::pair<matrix<double>,matrix<double> > qrDecompose(matrix<double> a){
    matrix<double> q=a;
    q=q.unitize();
    matrix<double> qT(q.m,q.n);
    for(int i=0;i<q.n;i++){
        for(int j=0;j<q.m;j++){
            qT.mat[j][i]=q.mat[i][j];
        }
    }
    matrix<double> r=qT*a;
    /*
    std::cout<<"q:"<<std::endl;
    q.putMatrix();
    std::cout<<"qT:"<<std::endl;
    qT.putMatrix();
    std::cout<<"a:"<<std::endl;
    a.putMatrix();
    std::cout<<"r:"<<std::endl;
    r.putMatrix();
    std::cout<<std::endl;
    */
    return std::make_pair(q,r);
}
std::vector<double> eig(matrix<double> a,int precision=1000){
    for(int i=0;i<precision;i++){
        /*
        std::cout<<"a in eig:"<<std::endl;
        a.putMatrix();
        */
        std::pair<matrix<double>,matrix<double> > pa=qrDecompose(a);
        matrix<double> a1=pa.second*pa.first;
        a=a1;
    }
    std::vector<double> eigv(a.n,0);
    for(int i=0;i<a.n;i++){
        eigv[i]=a.mat[i][i];
    }
    return eigv;
}
template<typename T>
inline T detRemainder(matrix<T>& M){
    if(M.n==1){
        return M.mat[0][0];
    }
    T ans;
    ans.init(0,1);
    for(int i=0;i<M.m;i++){
        matrix<T> A;
        A.init(M.n-1,M.m-1);
        A=M.getRemainder(0,i);
        T f=M.mat[0][i];
        T g=detRemainder(A);
        if(i&1){
            ans=ans-M.mat[0][i]*detRemainder(A);
        }
        else{
            ans=ans+M.mat[0][i]*detRemainder(A);
        }
    }
    return ans;
}
int main(){
    /*
    std::vector<frac> p;
    p.push_back(frac(1,2));
    p.push_back(frac(-2,3));
    p.push_back(frac(1,1));
    std::vector<double> pp=unitize(p);
    std::cout<<pp[0]<<" "<<pp[1]<<" "<<pp[2]<<std::endl;
    */
    std::vector<matrix<frac> > v;
    matrix<frac> garbage;
    v.push_back(garbage);
    std::cout<<"这是矩阵计算器。你可以输入'Input'来录入矩阵，'Change'来修改矩阵，'Output'来输出矩阵，'Multiply'来计算矩阵乘法，'Scalar'来对矩阵进行数乘，'Add'来执行矩阵加法，'Subtract'来执行矩阵减法，'Power'来计算矩阵的幂，'Det'来计算矩阵的行列式，'Inverse'来计算逆矩阵，'Orthogonalize'来正交化矩阵的各列向量，'Unitize'来单位化，'Eig'来计算特征值，或输入'Solve'来解线性方程组。"<<std::endl;
    std::cout<<"本程序支持分数。输入'End'退出。"<<std::endl;
    while(1){
        std::string s;
        getline(std::cin,s);
        if(s=="Input"||s=="input"){
            std::cout<<"请输入矩阵的行数和列数："<<std::endl;
            int n,m;
            std::cin>>n>>m;
            matrix<frac> a;
            a.init(n,m);
            std::cout<<"请输入矩阵："<<std::endl;
            char ch=getchar();
            a.readMatrix(n,m);
            v.push_back(a);
            std::cout<<"录入成功，本矩阵编号为"<<v.size()-1<<std::endl;
        }
        if(s=="Change"||s=="change"){
            std::cout<<"请输入要修改的矩阵编号："<<std::endl;
            int k;
            std::cin>>k;
            if(k<1||k>=v.size()){
                std::cout<<"无此编号。"<<std::endl;
                continue;
            }
            std::cout<<"请输入矩阵的行数和列数："<<std::endl;
            int n,m;
            std::cin>>n>>m;
            matrix<frac> a;
            a.init(n,m);
            std::cout<<"请输入矩阵："<<std::endl;
            char ch=getchar();
            a.readMatrix(n,m);
            v[k]=a;
        }
        if(s=="Output"||s=="output"){
            std::cout<<"请输入要输出的矩阵编号："<<std::endl;
            int k;
            std::cin>>k;
            if(k<1||k>=v.size()){
                std::cout<<"无此编号。"<<std::endl;
                continue;
            }
            v[k].putMatrix();
        }
        if(s=="Multiply"||s=="multiply"){
            std::cout<<"请输入要相乘的两个矩阵编号："<<std::endl;
            int k,r;
            std::cin>>k>>r;
            if(k<1||k>=v.size()||r<1||r>=v.size()){
                std::cout<<"无此编号。"<<std::endl;
                continue;
            }
            if(v[k].m!=v[r].n){
                std::cout<<"矩阵大小不一致。"<<std::endl;
                continue;
            }
            matrix<frac> a=v[k]*v[r];
            std::cout<<"乘积为："<<std::endl;
            a.putMatrix();
            std::cout<<"已自动将矩阵存储，编号为"<<v.size()<<std::endl;
            v.push_back(a);
        }
        if(s=="Scalar"||s=="scalar"){
            std::cout<<"请输入要数乘的矩阵编号："<<std::endl;
            int k;
            std::cin>>k;
            if(k<1||k>=v.size()){
                std::cout<<"无此编号。"<<std::endl;
                continue;
            }
            frac f;
            std::cout<<"请输入要乘以的数："<<std::endl;
            char ch=getchar();
            f.readFrac();
            matrix<frac> a=v[k]*f;
            std::cout<<"乘积为："<<std::endl;
            a.putMatrix();
            std::cout<<"已自动将矩阵存储，编号为"<<v.size()<<std::endl;
            v.push_back(a);
        }
        if(s=="Add"||s=="add"){
            std::cout<<"请输入要相加的两个矩阵编号："<<std::endl;
            int k,r;
            std::cin>>k>>r;
            if(k<1||k>=v.size()||r<1||r>=v.size()){
                std::cout<<"无此编号。"<<std::endl;
                continue;
            }
            if(v[k].n!=v[r].n||v[k].m!=v[r].m){
                std::cout<<"矩阵大小不一致。"<<std::endl;
                continue;
            }
            matrix<frac> a=v[k]+v[r];
            std::cout<<"和为："<<std::endl;
            a.putMatrix();
            std::cout<<"已自动将矩阵存储，编号为"<<v.size()<<std::endl;
            v.push_back(a);
        }
        if(s=="Subtract"||s=="subtract"){
            std::cout<<"请输入要相减的两个矩阵编号："<<std::endl;
            int k,r;
            std::cin>>k>>r;
            if(k<1||k>=v.size()||r<1||r>=v.size()){
                std::cout<<"无此编号。"<<std::endl;
                continue;
            }
            if(v[k].n!=v[r].n||v[k].m!=v[r].m){
                std::cout<<"矩阵大小不一致。"<<std::endl;
                continue;
            }
            matrix<frac> a=v[k]-v[r];
            std::cout<<"差为："<<std::endl;
            a.putMatrix();
            std::cout<<"已自动将矩阵存储，编号为"<<v.size()<<std::endl;
            v.push_back(a);
        }
        if(s=="Power"||s=="power"){
            std::cout<<"请输入要计算幂的矩阵编号："<<std::endl;
            int k;
            std::cin>>k;
            if(k<1||k>=v.size()){
                std::cout<<"无此编号。"<<std::endl;
                continue;
            }
            if(v[k].m!=v[k].n){
                std::cout<<"不是方阵。"<<std::endl;
                continue;
            }
            int p;
            std::cout<<"请输入幂次："<<std::endl;
            std::cin>>p;
            if(p<0){
                std::cout<<"幂次错误。"<<std::endl;
                continue;
            }
            matrix<frac> a=v[k].quickPower(p);
            std::cout<<"幂为："<<std::endl;
            a.putMatrix();
            std::cout<<"已自动将矩阵存储，编号为"<<v.size()<<std::endl;
            v.push_back(a);
        }
        if(s=="Det"||s=="det"){
            std::cout<<"请输入要计算行列式的矩阵编号："<<std::endl;
            int k;
            std::cin>>k;
            if(k<1||k>=v.size()){
                std::cout<<"无此编号。"<<std::endl;
                continue;
            }
            if(v[k].m!=v[k].n){
                std::cout<<"不是方阵。"<<std::endl;
                continue;
            }
            frac f=v[k].det();
            std::cout<<"行列式为："<<std::endl;
            f.putFrac();
            std::cout<<std::endl;
        }
        if(s=="Solve"||s=="solve"){
            std::cout<<"请输入增广矩阵的编号："<<std::endl;
            int k;
            std::cin>>k;
            if(k<1||k>=v.size()){
                std::cout<<"无此编号。"<<std::endl;
                continue;
            }
            matrix<frac> a;
            a.copy(v[k]);
            a.putMatrix();
            a.gaussElimination();
            int op=a.getVerdict();
            if(!op){
                std::cout<<"存在唯一解："<<std::endl;
                a.putMatrix();
                std::cout<<"已自动将矩阵存储，编号为"<<v.size()<<std::endl;
                v.push_back(a);
            }
            else if(op==-1){
                std::cout<<"无解。"<<std::endl;
            }
            else{
                a.putMatrix();
                std::cout<<"无数解。"<<std::endl;
                matrix<frac> b=a.getSolveSet(op);
                std::cout<<"解为"<<std::endl;
                b.putSolveSet();
                std::cout<<"已自动将矩阵存储，编号为"<<v.size()<<std::endl;
                v.push_back(b);
            }
        }
        if(s=="Inverse"||s=="inverse"){
            std::cout<<"请输入要求逆的矩阵编号："<<std::endl;
            int k;
            std::cin>>k;
            matrix<frac> a;
            a.copy(v[k]);
            frac z;
            z.init(0,1);
            if(a.det()!=z){
                matrix<frac> r=a.inverse();
                r.putMatrix();
                std::cout<<"已自动将矩阵存储，编号为"<<v.size()<<std::endl;
                v.push_back(r);
            }
            else{
                std::cout<<"该矩阵不可逆。"<<std::endl;
            }
        }
        if(s=="Orthogonalize"||s=="orthogonalize"){
            std::cout<<"请输入要正交化的矩阵编号："<<std::endl;
            int k;
            std::cin>>k;
            matrix<frac> a;
            a.copy(v[k]);
            a.orthogonalize();
            std::cout<<"正交化的矩阵为："<<std::endl;
            a.putMatrix();
            std::cout<<"已自动将矩阵存储，编号为"<<v.size()<<std::endl;
            v.push_back(a);
        }
        if(s=="Unitize"||s=="unitize"){
            std::cout<<"请输入要单位化的矩阵编号："<<std::endl;
            int k;
            std::cin>>k;
            matrix<frac> a;
            a.copy(v[k]);
            matrix<double> ra=a.unitize();
            std::cout<<"单位化的矩阵为："<<std::endl;
            ra.putMatrix();
        }
        if(s=="Eig"||s=="eig"){
            std::cout<<"请输入要计算特征值的矩阵编号："<<std::endl;
            int k;
            std::cin>>k;
            if(k<1||k>=v.size()){
                std::cout<<"无此编号。"<<std::endl;
                continue;
            }
            if(v[k].m!=v[k].n){
                std::cout<<"不是方阵。"<<std::endl;
                continue;
            }
            matrix<double> a(v[k].n,v[k].m);
            for(int i=0;i<v[k].n;i++){
                for(int j=0;j<v[k].m;j++){
                    a.mat[i][j]=v[k].mat[i][j].real();
                }
            }
            std::vector<double> e=eig(a);
            for(int i=0;i<e.size();i++){
                std::cout<<e[i]<<" ";
            }
            std::cout<<std::endl;
        }
        if(s=="End"||s=="end"){
            return 0;
        }
    }
    return 0;
}