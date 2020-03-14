clear
clc
syms x y z;
ux = 0.1*x*x;
uy = 0;
uz = 0;

du = [diff(ux, x), diff(ux, y), diff(ux, z);
    diff(uy, x), diff(uy, y), diff(uz, z);
    diff(uz, x), diff(uz, y), diff(uz, z)];

F = du + eye(3);

C = transpose(F)*F;

% c11, c12 ... are C's entries
c11 = C(1,1);
c12 = C(1,2);
c13 = C(1,3);
c21 = C(2,1);
c22 = C(2,2);
c23 = C(2,3);
c31 = C(3,1);
c32 = C(3,2);
c33 = C(3,3);

I1 = trace(C);
I2 = 1/2*(trace(C)*trace(C) - trace(C*C));
I3 = det(C);

A_1 = [cos(pi/12), -sin(pi/12), 0];
A_2 = [cos(pi/12), sin(pi/12), 0];

M_1 = zeros(3,3);

for x=1:3
    for y=1:3
        M_1(x,y) = A_1(x)*A_1(y);
    end
end

M_2 = zeros(3,3);

for x=1:3
    for y=1:3
        M_2(x,y) = A_2(x)*A_2(y);
    end
end

J4_1 = trace(C*M_1);
%J4_2 = trace(C*M_2);


eta1 = 141;
eta2 = 160;
eta3 = 3100;
delta = 2*eta1 + 4*eta2 + 2*eta3;

e1 = 0.1;
e2 = 1;

k1 = 6.85;
k2 = 754.01;


dI3 =[ c22*c33 - c23*c32, c23*c31 - c21*c33, c21*c32 - c22*c31;
    c13*c32 - c12*c33, c11*c33 - c13*c31, c12*c31 - c11*c32;
    c12*c23 - c13*c22, c13*c21 - c11*c23, c11*c22 - c12*c21];

dI2 = [c22 + c33, -c21, -c31;
    -c12, c11 + c33, -c32;
    -c13, -c23, c11 + c22];

dI1 = eye(3);

dJ4_1 = [ 4201915656573739/4503599627370496, -1/4, 0;
    -1/4, 4826943532748117/72057594037927936, 0;
    0, 0, 0];

%% iso
psi_MR = eta1*I1 + eta2*I2 + eta3*I3 - delta*log(sqrt(I3));

S_MR = 2*(eta1*dI1 + eta2*dI2 + eta3*dI3 - delta/2/I3*dI3);
P_MR = inv(F)*S_MR;

bx_MR = -diff(P_MR(1,1), x) - diff(P_MR(1,2),y) - diff(P_MR(1,3),z);

by_MR = -diff(P_MR(2,1), x) - diff(P_MR(2,2),y) - diff(P_MR(2,3),z);

bz_MR = -diff(P_MR(3,1), x) - diff(P_MR(3,2),y) - diff(P_MR(3,3),z);

%% penalty

S_P = 2*e1*(e2*I3^(e2-1)*dI3 - e2*I3^(-e2-1)*dI3);

P_P = inv(F)*S_P;

bx_P = -diff(P_P(1,1), x) - diff(P_P(1,2),y) - diff(P_P(1,3),z);

by_P = -diff(P_P(2,1), x) - diff(P_P(2,2),y) - diff(P_P(2,3),z);

bz_P = -diff(P_P(3,1), x) - diff(P_P(3,2),y) - diff(P_P(3,3),z);


%% tissue

% J4 < 0

%S_T_1 = 0;
%S_T_2 = 0;

%J4 > 0

S_T_1 = k1/2/k2*exp(k2*(J4_1-1)^2)*k2*2*(J4_1-1)*dJ4_1;
S_T_2 = k1/2/k2*exp(k2*(J4_1*J4_1-1)^2)*k2*2*(J4_1-1)*dJ4_1;

P_T_1 = inv(F)*S_T_1;
P_T_2 = inv(F)*S_T_2;

bx_T_1 = -diff(P_T_1(1,1), x) - diff(P_T_1(1,2),y) - diff(P_T_1(1,3),z);

by_T_1 = -diff(P_T_1(2,1), x) - diff(P_T_1(2,2),y) - diff(P_T_1(2,3),z);

bz_T_1 = -diff(P_T_1(3,1), x) - diff(P_T_1(3,2),y) - diff(P_T_1(3,3),z);


bx_T_2 = -diff(P_T_2(1,1), x) - diff(P_T_2(1,2),y) - diff(P_T_2(1,3),z);

by_T_2 = -diff(P_T_2(2,1), x) - diff(P_T_2(2,2),y) - diff(P_T_2(2,3),z);

bz_T_2 = -diff(P_T_2(3,1), x) - diff(P_T_2(3,2),y) - diff(P_T_2(3,3),z);
