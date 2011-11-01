%EXERCICE 2
%chargement des données
trainA = load('C:\Program Files\MATLAB\test\classificationA.mat', '-ascii') ;
trainB = load('C:\Program Files\MATLAB\test\classificationB.train.mat', '-ascii') ;
trainC = load('C:\Program Files\MATLAB\test\classificationC.train.mat', '-ascii') ;
testA = load('C:\Program Files\MATLAB\test\classificationA.test.mat', '-ascii') ;
testB = load('C:\Program Files\MATLAB\test\classificationB.test.mat', '-ascii') ;
testC = load('C:\Program Files\MATLAB\test\classificationC.test.mat', '-ascii') ;

%Question 1

%Maximum de vraisemblance
Y=trainA(:,3)  %Y est la colonne avec les valeurs de y
s=sum(Y)             %y est le nombre de données pour lesquelles y=1

Z=1.-trainA(:,3)  %Z est la colonne avec les valeurs 1-y

X=trainA(:,1:2)'  %X est la matrice 2x300 des valeurs de x


pi = s/300
m=(X*Y)/s           %m est mu1
n=(X*Z)/(300-s)     %n est mu0



%calcul de Sigma

W=zeros(300,2)    %W est la matrice de taille 300x2 dont les deux colonnes sont Y
T=zeros(300,2)    %T est la matrice de taille 300x2 dont les deux colonnes sont Z
W(:,1)=Y(:,1)
W(:,2)=Y(:,1)
T(:,1)=Z(:,1)
T(:,2)=Z(:,1)

M=zeros(300,2)      %M est la matrice de taille 300x2 dont les lignes sont la transposée de m
N=zeros(300,2)      %N est la matrice de taille 300x2 dont les lignes sont la transposée de n

M(:,1)=m(1,1)
M(:,2)=m(2,1)
N(:,1)=n(1,1)
N(:,2)=n(2,1)


pi
m
n

Sigma = ((X'-M)'*(W.*(X'-M)) + (X'-N)'*(T.*(X'-N)))/300


 abscisse=zeros(s,1)
 Abscisse=zeros(300-s,1)
 ordonnee=zeros(s,1)
 Ordonnee=zeros(300-s,1)

j=1
k=1
for i=1:300
    if Y(i,1)==1
        abscisse(j,1)=X(1,i)
        ordonnee(j,1)=X(2,i)
        j=j+1
    else 
        Abscisse(k,1)=X(1,i)
        Ordonnee(k,1)=X(2,i)
        k=k+1
    end
end
psi=inv(Sigma) 
a=psi*(m-n)
b=log((1-pi)/pi)+0.5.*n'*psi*n-0.5.*m'*psi*m


%a =
%
 %  -1.4335
  %  1.7390


%b =

 %  -2.6690
 
hold on
scatter(abscisse,ordonnee,'.','red')
scatter(Abscisse,Ordonnee,'.','blue')
h=ezplot('-1.4335*x+1.7390*y-2.6690')
set(h,'Color','m')                             %la droite p(y=1|x)=0.5 pour le modèle LDA est magenta, son équation est a.[x,y]+b=0
hold off
 

%Question 2
theta = zeros(3,1)
U=zeros(3,300)              %U est la matrice 3x300 où les deux premières lignes sont formées de X et la dernière a toutes ses valeurs égales à 1
           
U(1,:)=X(1,:)
U(2,:)=X(2,:)
U(3,:)=1

mu=(1./(1.+exp(-theta'*U)))' %mu est la colonne avec les valeurs de mu déduites de la données X et de theta



W=diag(mu.*(1.-mu))        %W est la matrice de taille 300x300 diagonale dont les éléments diagonaux sont les mu(1-mu)


l=1
d=1

while l<500 && d>10^(-10)
    m=(inv(U*W*(U')))*U*(Y-mu)
    d= m'*m
    theta=theta+m
    mu=(1./(1.+exp(-theta'*U)))'
    W=diag(mu.*(1.-mu))
end

theta



%theta =

 %  -1.3452
  %  1.6739
   %-1.5605
   
   

 hold on
scatter(abscisse,ordonnee,'.','red')
scatter(Abscisse,Ordonnee,'.','blue')
ezplot('-1.3452*x+1.6739*y-1.5605')                   %tracer la droite telle que p(y=1 |x) = 0.5 ie tel que [x,y,1]*theta=0, elle est verte, son équation est theta.[x,y,1]=0
hold off 


 %Question 3
 
 %équation normale : X*(X')*theta = X*Y
 theta=inv(U*(U'))*U*Y                   %afin d'avoir l'équation affine directement (et non seulement linéaire), on utilise U au lieu de X 

 %theta =

%   -0.1756
 %   0.2130
  %  0.3066
 
 
l=ezplot('-0.1756*x+0.213*y+0.3066-0.5')   %%tracer la droite telle que p(y=1 |x) = 0.5 ie tel que [x,y,1]*theta=0.5, elle est noire
set(l,'Color','black')
 




 %Question 4
 % Le taux d'erreur est la proportion de données qui ont été mal classées
 %Pour classer les données dans les modèles LDA et IRLS, on étudie p(y=1 |x) : si p(y=1 |x)<0.5 alors
 %y=0, sinon y=1. Pour la régression linéaire on regarde theta*u (avec u le
 %vecteur(x1,x2,1) )

 A=testA(:,3)         %A correspond à la dernière colonne de testA
 
 LDA=zeros(300,1)
 IRLS=zeros(300,1)
 RL=zeros(300,1)
 
 for i=1:300
     if -1.4335*testA(i,1)+1.7390*testA(i,2)-2.6690 > 0
  
        LDA(i,1)=1
     end
 end
 
  for i=1:300
     if -1.3452*testA(i,1)+1.6739*testA(i,2)-1.5605 > 0
         IRLS(i,1)=1
     end
  end
 
  for i=1:300
     if  -0.1756*testA(i,1)+0.213*testA(i,2)+0.3066>0.5
         RL(i,1)=1
     end
  end
 
 errLDA=mean(abs(LDA-A))
 errIRLS=mean(abs(IRLS-A))
 errRL=mean(abs(RL-A))
 
% errLDA =

 %   0.2167


%errIRLS =

 %   0.1600


%errRL =

 %   0.1600
 
 
 
 %Question 5
 
Y=trainA(:,3)  %Y est la colonne avec les valeurs de y
s=sum(Y)             %y est le nombre de données pour lesquelles y=1

Z=1.-trainA(:,3)  %Z est la colonne avec les valeurs 1-y

X=trainA(:,1:2)'  %X est la matrice 2x300 des valeurs de x


pi = s/300
m=(X*Y)/s           %m est mu1
n=(X*Z)/(300-s)     %n est mu0




%calcul de Sigma0 et Sigma1

W=zeros(300,2)    %W est la matrice de taille 300x2 dont les deux colonnes sont Y
T=zeros(300,2)    %T est la matrice de taille 300x2 dont les deux colonnes sont Z
W(:,1)=Y(:,1)
W(:,2)=Y(:,1)
T(:,1)=Z(:,1)
T(:,2)=Z(:,1)

M=zeros(300,2)      %M est la matrice de taille 300x2 dont les lignes sont la transposée de m
N=zeros(300,2)      %N est la matrice de taille 300x2 dont les lignes sont la transposée de n

M(:,1)=m(1,1)
M(:,2)=m(2,1)
N(:,1)=n(1,1)
N(:,2)=n(2,1)


pi
m
n

Sigma =  (X'-N)'*(T.*(X'-N))/(300-s)     %Sigma est sigma0
sigma = (X'-M)'*(W.*(X'-M))/s            %Sigma est sigma1


%pi =

 %   0.6300


%m =

 %  -1.0392
  %  1.4857


%n =

 %  -0.1648
  % -0.0207


%Sigma =

 %   1.0710    0.3117
  %  0.3117    1.2277


%sigma =

 %   0.8622    0.2481
  %  0.2481    1.0095
  
  
%Tracé de la conique d'équation p(Y=1|x)=0.5  
  
 d=sigma-Sigma

%d =

%   -0.2088   -0.0635
 %  -0.0635   -0.2183
 
 
%  n'*inv(Sigma)-m'*inv(sigma)

%ans =

 %   1.5919   -1.8787

%inv(Sigma)*n-inv(sigma)*m

%ans =

 %   1.5919
  % -1.8787

 
 
 % b=(-1.8787+ -1.8787)/2

%b =

 %  -1.8787

%a=( 1.5919+ 1.5919)/2

%a =

 %   1.5919

%c=(log(sqrt(det(sigma))*(1-pi)/(sqrt(det(Sigma))*pi)))+m'*sigma*m+n'*Sigma*n

%c =

   % 8.1620


 g= ezplot('-0.2088*x^2-0.2183*y^2-2*0.0635*x*y+1.5919*x-1.8787*y+1.6881')  %%tracer la droite telle que p(y=1 |x) = 0.5 ie tel que [x,y]\Sigma[x,y] + ax+by+c = 0, elle est bleue
 set(g,'Color','blue')

 
 
  %Figure avec les 4 méthodes : 
hold on
scatter(abscisse,ordonnee,'.','red')
scatter(Abscisse,Ordonnee,'.','blue')
h=ezplot('-1.4335*x+1.7390*y-2.6690')
set(h,'Color','m')                   %la droite p(y=1|x)=0.5 pour le modèle LDA est magenta
ezplot('-1.3452*x+1.6739*y-1.5605')  %la droite p(y=1|x)=0.5 pour le modèle IRLS est verte
l=ezplot('-0.1756*x+0.213*y+0.3066-0.5')
set(l,'Color','black')               %la droite p(y=1|x)=0.5 pour la régression linéaire est noire
g= ezplot('-0.2088*x^2-0.2183*y^2-2*0.0635*x*y+1.5919*x-1.8787*y+1.6881')
set(g,'Color','blue')
hold off
 
 
 %Tracé pour testA
 X=testA(:,1:2)'

 Y=testA(:,3)  %Y est la colonne avec les valeurs de y
s=sum(Y)             %y est le nombre de données pour lesquelles y=1

 
 abscisse=zeros(s,1)
 Abscisse=zeros(300-s,1)
 ordonnee=zeros(s,1)
 Ordonnee=zeros(300-s,1)

j=1
k=1
for i=1:300
    if Y(i,1)==1
        abscisse(j,1)=X(1,i)
        ordonnee(j,1)=X(2,i)
        j=j+1
    else 
        Abscisse(k,1)=X(1,i)
        Ordonnee(k,1)=X(2,i)
        k=k+1
    end
end
 
  %Figure avec les 4 méthodes (test) : 
hold on
scatter(abscisse,ordonnee,'.','red')
scatter(Abscisse,Ordonnee,'.','blue')
h=ezplot('-1.4335*x+1.7390*y-2.6690')
set(h,'Color','m')                   %la droite p(y=1|x)=0.5 pour le modèle LDA est magenta
ezplot('-1.3452*x+1.6739*y-1.5605')  %la droite p(y=1|x)=0.5 pour le modèle IRLS est verte
l=ezplot('-0.1756*x+0.213*y+0.3066-0.5')
set(l,'Color','black')               %la droite p(y=1|x)=0.5 pour la régression linéaire est noire
g= ezplot('-0.2088*x^2-0.2183*y^2-2*0.0635*x*y+1.5919*x-1.8787*y+1.6881')
set(g,'Color','blue')
hold off

 
 