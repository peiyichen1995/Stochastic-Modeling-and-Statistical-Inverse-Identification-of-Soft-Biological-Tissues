clear
clc
syms x y z;
ux = 0.1*x*x;
uy = 0.1*y*y;

du = [diff(ux, x), diff(ux, y);
    diff(uy, x), diff(uy, y);];

F = du + eye(2);

C = transpose(F)*F;
%%
% c11, c12 ... are C's entries
c11 = C(1,1);
c12 = C(1,2);
c21 = C(2,1);
c22 = C(2,2);

I1 = trace(C);
I2 = 1/2*(trace(C)*trace(C) - trace(C*C));
I3 = det(C);

dI3 = [c22, -c21;
    -c12, c11];
dI2 = dI3;
dI1 = eye(2);

eta1 = 141;
eta2 = 160;
eta3 = 3100;
delta = 2*eta1 + 4*eta2 + 2*eta3;

e1 = 0.005;
e2 = 10;
%% iso

cof = [c22, -c12;
    -c21, c11];

S_MR = 2*(eta1*dI1 + eta2*dI2 + eta3*dI3 - delta/2/I3*dI3);
%S_MR = 2*(eta1 + eta2*I1)*eye(2) - eta2*C + (eta3 - delta/2/I3)*cof;
P_MR = F*S_MR;

bx_MR = -diff(P_MR(1,1), x) - diff(P_MR(1,2),y);

by_MR = -diff(P_MR(2,1), x) - diff(P_MR(2,2),y);

%% penalty
S_P = 2*e1*(e2*I3^(e2-1)*dI3 - e2*I3^(-e2-1)*dI3);
%S_P = 2*e1*e2*(I3^(e2-1) - I3^(-e2-1))*cof;

P_P = F*S_P;

bx_P = -diff(P_P(1,1), x) - diff(P_P(1,2),y);

by_P = -diff(P_P(2,1), x) - diff(P_P(2,2),y);

%% tissue
A_1 = [sqrt(0.5), sqrt(0.5)];

M_1 = zeros(2,2);

for x=1:2
    for y=1:2
        M_1(x,y) = A_1(x)*A_1(y);
    end
end

k1 = 0.1;
k2 = 0.04;

J4_1 = trace(C*M_1);

dJ4_1 = [0.5, 0.5;
    0.5, 0.5];


%S_T_1 = 2*k1/2/k2*exp(k2*(J4_1-1)^2)*k2*2*(J4_1-1)*dJ4_1;
S_T_1 = k1/2/k2*exp(k2*(J4_1-1)^2)*k2*2*(2*(J4_1-1)^2-(J4_1)^3)*(4*(J4_1-1)-3*(J4_1-1)^2)*dJ4_1;
S_T_2 = k1/2/k2*exp(k2*(J4_1*J4_1-1)^2)*k2*2*(J4_1-1)*dJ4_1;

P_T_1 = F*S_T_1;
P_T_2 = F*S_T_2;

bx_T_1 = -diff(P_T_1(1,1), x) - diff(P_T_1(1,2),y);

by_T_1 = -diff(P_T_1(2,1), x) - diff(P_T_1(2,2),y);



bx_T_2 = -diff(P_T_2(1,1), x) - diff(P_T_2(1,2),y);

by_T_2 = -diff(P_T_2(2,1), x) - diff(P_T_2(2,2),y);


