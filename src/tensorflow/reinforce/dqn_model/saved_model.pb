¤©
Ò¶
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02unknown8Ó

dqn_model/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_namedqn_model/dense/kernel

*dqn_model/dense/kernel/Read/ReadVariableOpReadVariableOpdqn_model/dense/kernel*
_output_shapes

:@*
dtype0

dqn_model/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_namedqn_model/dense/bias
y
(dqn_model/dense/bias/Read/ReadVariableOpReadVariableOpdqn_model/dense/bias*
_output_shapes
:@*
dtype0

dqn_model/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *)
shared_namedqn_model/dense_1/kernel

,dqn_model/dense_1/kernel/Read/ReadVariableOpReadVariableOpdqn_model/dense_1/kernel*
_output_shapes

:@ *
dtype0

dqn_model/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namedqn_model/dense_1/bias
}
*dqn_model/dense_1/bias/Read/ReadVariableOpReadVariableOpdqn_model/dense_1/bias*
_output_shapes
: *
dtype0

dqn_model/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *)
shared_namedqn_model/dense_2/kernel

,dqn_model/dense_2/kernel/Read/ReadVariableOpReadVariableOpdqn_model/dense_2/kernel*
_output_shapes

:  *
dtype0

dqn_model/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namedqn_model/dense_2/bias
}
*dqn_model/dense_2/bias/Read/ReadVariableOpReadVariableOpdqn_model/dense_2/bias*
_output_shapes
: *
dtype0

dqn_model/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *)
shared_namedqn_model/dense_3/kernel

,dqn_model/dense_3/kernel/Read/ReadVariableOpReadVariableOpdqn_model/dense_3/kernel*
_output_shapes

: *
dtype0

dqn_model/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namedqn_model/dense_3/bias
}
*dqn_model/dense_3/bias/Read/ReadVariableOpReadVariableOpdqn_model/dense_3/bias*
_output_shapes
:*
dtype0

dqn_model/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *)
shared_namedqn_model/dense_4/kernel

,dqn_model/dense_4/kernel/Read/ReadVariableOpReadVariableOpdqn_model/dense_4/kernel*
_output_shapes

:  *
dtype0

dqn_model/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namedqn_model/dense_4/bias
}
*dqn_model/dense_4/bias/Read/ReadVariableOpReadVariableOpdqn_model/dense_4/bias*
_output_shapes
: *
dtype0

dqn_model/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *)
shared_namedqn_model/dense_5/kernel

,dqn_model/dense_5/kernel/Read/ReadVariableOpReadVariableOpdqn_model/dense_5/kernel*
_output_shapes

: *
dtype0

dqn_model/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namedqn_model/dense_5/bias
}
*dqn_model/dense_5/bias/Read/ReadVariableOpReadVariableOpdqn_model/dense_5/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Adam/dqn_model/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*.
shared_nameAdam/dqn_model/dense/kernel/m

1Adam/dqn_model/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dqn_model/dense/kernel/m*
_output_shapes

:@*
dtype0

Adam/dqn_model/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameAdam/dqn_model/dense/bias/m

/Adam/dqn_model/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dqn_model/dense/bias/m*
_output_shapes
:@*
dtype0

Adam/dqn_model/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *0
shared_name!Adam/dqn_model/dense_1/kernel/m

3Adam/dqn_model/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dqn_model/dense_1/kernel/m*
_output_shapes

:@ *
dtype0

Adam/dqn_model/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameAdam/dqn_model/dense_1/bias/m

1Adam/dqn_model/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dqn_model/dense_1/bias/m*
_output_shapes
: *
dtype0

Adam/dqn_model/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *0
shared_name!Adam/dqn_model/dense_3/kernel/m

3Adam/dqn_model/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dqn_model/dense_3/kernel/m*
_output_shapes

: *
dtype0

Adam/dqn_model/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/dqn_model/dense_3/bias/m

1Adam/dqn_model/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dqn_model/dense_3/bias/m*
_output_shapes
:*
dtype0

Adam/dqn_model/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *0
shared_name!Adam/dqn_model/dense_5/kernel/m

3Adam/dqn_model/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dqn_model/dense_5/kernel/m*
_output_shapes

: *
dtype0

Adam/dqn_model/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/dqn_model/dense_5/bias/m

1Adam/dqn_model/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dqn_model/dense_5/bias/m*
_output_shapes
:*
dtype0

Adam/dqn_model/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*.
shared_nameAdam/dqn_model/dense/kernel/v

1Adam/dqn_model/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dqn_model/dense/kernel/v*
_output_shapes

:@*
dtype0

Adam/dqn_model/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameAdam/dqn_model/dense/bias/v

/Adam/dqn_model/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dqn_model/dense/bias/v*
_output_shapes
:@*
dtype0

Adam/dqn_model/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *0
shared_name!Adam/dqn_model/dense_1/kernel/v

3Adam/dqn_model/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dqn_model/dense_1/kernel/v*
_output_shapes

:@ *
dtype0

Adam/dqn_model/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameAdam/dqn_model/dense_1/bias/v

1Adam/dqn_model/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dqn_model/dense_1/bias/v*
_output_shapes
: *
dtype0

Adam/dqn_model/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *0
shared_name!Adam/dqn_model/dense_3/kernel/v

3Adam/dqn_model/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dqn_model/dense_3/kernel/v*
_output_shapes

: *
dtype0

Adam/dqn_model/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/dqn_model/dense_3/bias/v

1Adam/dqn_model/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dqn_model/dense_3/bias/v*
_output_shapes
:*
dtype0

Adam/dqn_model/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *0
shared_name!Adam/dqn_model/dense_5/kernel/v

3Adam/dqn_model/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dqn_model/dense_5/kernel/v*
_output_shapes

: *
dtype0

Adam/dqn_model/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/dqn_model/dense_5/bias/v

1Adam/dqn_model/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dqn_model/dense_5/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ãI
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*I
valueIBI BI
Ü

dense1

dense2
	adv_dense
adv_out
v_dense
	v_out
lambda_layer
combine
		optimizer


signatures
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature*
Ë

kernel
bias
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
Ë

kernel
bias
#_self_saveable_object_factories
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses*
Ë

%kernel
&bias
#'_self_saveable_object_factories
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses*
Ë

.kernel
/bias
#0_self_saveable_object_factories
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses*
Ë

7kernel
8bias
#9_self_saveable_object_factories
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses*
Ë

@kernel
Abias
#B_self_saveable_object_factories
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses*
³
#I_self_saveable_object_factories
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses* 
³
#P_self_saveable_object_factories
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses* 
ä
Witer

Xbeta_1

Ybeta_2
	Zdecay
[learning_ratemmmm.m/m@mAmvvvv.v/v@vAv*

\serving_default* 
* 
Z
0
1
2
3
%4
&5
.6
/7
78
89
@10
A11*
Z
0
1
2
3
%4
&5
.6
/7
78
89
@10
A11*
* 
°
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
XR
VARIABLE_VALUEdqn_model/dense/kernel(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEdqn_model/dense/bias&dense1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

0
1*
* 

bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
ZT
VARIABLE_VALUEdqn_model/dense_1/kernel(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEdqn_model/dense_1/bias&dense2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

0
1*
* 

gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*
* 
* 
]W
VARIABLE_VALUEdqn_model/dense_2/kernel+adv_dense/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEdqn_model/dense_2/bias)adv_dense/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

%0
&1*

%0
&1*
* 

lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*
* 
* 
[U
VARIABLE_VALUEdqn_model/dense_3/kernel)adv_out/kernel/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEdqn_model/dense_3/bias'adv_out/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

.0
/1*

.0
/1*
* 

qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*
* 
* 
[U
VARIABLE_VALUEdqn_model/dense_4/kernel)v_dense/kernel/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEdqn_model/dense_4/bias'v_dense/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

70
81*

70
81*
* 

vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*
* 
* 
YS
VARIABLE_VALUEdqn_model/dense_5/kernel'v_out/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEdqn_model/dense_5/bias%v_out/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

@0
A1*

@0
A1*
* 

{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
<
0
1
2
3
4
5
6
7*

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

total

count
	variables
	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
{u
VARIABLE_VALUEAdam/dqn_model/dense/kernel/mDdense1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/dqn_model/dense/bias/mBdense1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dqn_model/dense_1/kernel/mDdense2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/dqn_model/dense_1/bias/mBdense2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dqn_model/dense_3/kernel/mEadv_out/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/dqn_model/dense_3/bias/mCadv_out/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/dqn_model/dense_5/kernel/mCv_out/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/dqn_model/dense_5/bias/mAv_out/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dqn_model/dense/kernel/vDdense1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/dqn_model/dense/bias/vBdense1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dqn_model/dense_1/kernel/vDdense2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/dqn_model/dense_1/bias/vBdense2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dqn_model/dense_3/kernel/vEadv_out/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/dqn_model/dense_3/bias/vCadv_out/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/dqn_model/dense_5/kernel/vCv_out/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/dqn_model/dense_5/bias/vAv_out/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
serving_default_args_0Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
ó
StatefulPartitionedCallStatefulPartitionedCallserving_default_args_0dqn_model/dense/kerneldqn_model/dense/biasdqn_model/dense_1/kerneldqn_model/dense_1/biasdqn_model/dense_2/kerneldqn_model/dense_2/biasdqn_model/dense_3/kerneldqn_model/dense_3/biasdqn_model/dense_4/kerneldqn_model/dense_4/biasdqn_model/dense_5/kerneldqn_model/dense_5/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_10833
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ï
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*dqn_model/dense/kernel/Read/ReadVariableOp(dqn_model/dense/bias/Read/ReadVariableOp,dqn_model/dense_1/kernel/Read/ReadVariableOp*dqn_model/dense_1/bias/Read/ReadVariableOp,dqn_model/dense_2/kernel/Read/ReadVariableOp*dqn_model/dense_2/bias/Read/ReadVariableOp,dqn_model/dense_3/kernel/Read/ReadVariableOp*dqn_model/dense_3/bias/Read/ReadVariableOp,dqn_model/dense_4/kernel/Read/ReadVariableOp*dqn_model/dense_4/bias/Read/ReadVariableOp,dqn_model/dense_5/kernel/Read/ReadVariableOp*dqn_model/dense_5/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp1Adam/dqn_model/dense/kernel/m/Read/ReadVariableOp/Adam/dqn_model/dense/bias/m/Read/ReadVariableOp3Adam/dqn_model/dense_1/kernel/m/Read/ReadVariableOp1Adam/dqn_model/dense_1/bias/m/Read/ReadVariableOp3Adam/dqn_model/dense_3/kernel/m/Read/ReadVariableOp1Adam/dqn_model/dense_3/bias/m/Read/ReadVariableOp3Adam/dqn_model/dense_5/kernel/m/Read/ReadVariableOp1Adam/dqn_model/dense_5/bias/m/Read/ReadVariableOp1Adam/dqn_model/dense/kernel/v/Read/ReadVariableOp/Adam/dqn_model/dense/bias/v/Read/ReadVariableOp3Adam/dqn_model/dense_1/kernel/v/Read/ReadVariableOp1Adam/dqn_model/dense_1/bias/v/Read/ReadVariableOp3Adam/dqn_model/dense_3/kernel/v/Read/ReadVariableOp1Adam/dqn_model/dense_3/bias/v/Read/ReadVariableOp3Adam/dqn_model/dense_5/kernel/v/Read/ReadVariableOp1Adam/dqn_model/dense_5/bias/v/Read/ReadVariableOpConst*0
Tin)
'2%	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__traced_save_10961
®	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedqn_model/dense/kerneldqn_model/dense/biasdqn_model/dense_1/kerneldqn_model/dense_1/biasdqn_model/dense_2/kerneldqn_model/dense_2/biasdqn_model/dense_3/kerneldqn_model/dense_3/biasdqn_model/dense_4/kerneldqn_model/dense_4/biasdqn_model/dense_5/kerneldqn_model/dense_5/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dqn_model/dense/kernel/mAdam/dqn_model/dense/bias/mAdam/dqn_model/dense_1/kernel/mAdam/dqn_model/dense_1/bias/mAdam/dqn_model/dense_3/kernel/mAdam/dqn_model/dense_3/bias/mAdam/dqn_model/dense_5/kernel/mAdam/dqn_model/dense_5/bias/mAdam/dqn_model/dense/kernel/vAdam/dqn_model/dense/bias/vAdam/dqn_model/dense_1/kernel/vAdam/dqn_model/dense_1/bias/vAdam/dqn_model/dense_3/kernel/vAdam/dqn_model/dense_3/bias/vAdam/dqn_model/dense_5/kernel/vAdam/dqn_model/dense_5/bias/v*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_restore_11076ªë


ï
>__inference_dense_layer_call_and_return_conditional_losses_350

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê

¤
'__inference_dqn_model_layer_call_fn_796
input_1
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6:
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10:
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dqn_model_layer_call_and_return_conditional_losses_779`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1


ñ
@__inference_dense_1_layer_call_and_return_conditional_losses_666

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ê

¤
'__inference_dqn_model_layer_call_fn_731
input_1
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6:
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10:
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dqn_model_layer_call_and_return_conditional_losses_697`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ä
[
?__inference_lambda_layer_call_and_return_conditional_losses_382

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       E
MeanMeaninputsConst:output:0*
T0*
_output_shapes
: S
subSubinputsMean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
IdentityIdentitysub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±
f
<__inference_add_layer_call_and_return_conditional_losses_393

inputs
inputs_1
identityP
addAddV2inputsinputs_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í$

B__inference_dqn_model_layer_call_and_return_conditional_losses_697	
input

dense_4302:@

dense_4304:@
dense_1_4319:@ 
dense_1_4321: 
dense_2_4336:  
dense_2_4338: 
dense_3_4352: 
dense_3_4354:
dense_4_4369:  
dense_4_4371: 
dense_5_4385: 
dense_5_4387:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCallß
dense/StatefulPartitionedCallStatefulPartitionedCallinput
dense_4302
dense_4304*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_350
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_4319dense_1_4321*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_666
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_4336dense_2_4338*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_2_layer_call_and_return_conditional_losses_368
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_3_4352dense_3_4354*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_3_layer_call_and_return_conditional_losses_609
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_4_4369dense_4_4371*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_4_layer_call_and_return_conditional_losses_311
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_5_4385dense_5_4387*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_5_layer_call_and_return_conditional_losses_626Ö
lambda/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_lambda_layer_call_and_return_conditional_losses_382ò
add/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0lambda/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *E
f@R>
<__inference_add_layer_call_and_return_conditional_losses_393
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 k
IdentityIdentityadd/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
Ã	
ñ
@__inference_dense_5_layer_call_and_return_conditional_losses_626

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ä
[
?__inference_lambda_layer_call_and_return_conditional_losses_417

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       E
MeanMeaninputsConst:output:0*
T0*
_output_shapes
: S
subSubinputsMean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
IdentityIdentitysub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ï
 __inference__wrapped_model_10796

args_0!
dqn_model_10770:@
dqn_model_10772:@!
dqn_model_10774:@ 
dqn_model_10776: !
dqn_model_10778:  
dqn_model_10780: !
dqn_model_10782: 
dqn_model_10784:!
dqn_model_10786:  
dqn_model_10788: !
dqn_model_10790: 
dqn_model_10792:
identity¢!dqn_model/StatefulPartitionedCall
!dqn_model/StatefulPartitionedCallStatefulPartitionedCallargs_0dqn_model_10770dqn_model_10772dqn_model_10774dqn_model_10776dqn_model_10778dqn_model_10780dqn_model_10782dqn_model_10784dqn_model_10786dqn_model_10788dqn_model_10790dqn_model_10792*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_restored_function_body_10769y
IdentityIdentity*dqn_model/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
NoOpNoOp"^dqn_model/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2F
!dqn_model/StatefulPartitionedCall!dqn_model/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameargs_0
Æ

£
(__inference_restored_function_body_10769	
input
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6:
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10:
identity¢StatefulPartitionedCall½
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dqn_model_layer_call_and_return_conditional_losses_599o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
ó$

B__inference_dqn_model_layer_call_and_return_conditional_losses_837
input_1

dense_4646:@

dense_4648:@
dense_1_4651:@ 
dense_1_4653: 
dense_2_4656:  
dense_2_4658: 
dense_3_4661: 
dense_3_4663:
dense_4_4666:  
dense_4_4668: 
dense_5_4671: 
dense_5_4673:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCallá
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1
dense_4646
dense_4648*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_350
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_4651dense_1_4653*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_666
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_4656dense_2_4658*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_2_layer_call_and_return_conditional_losses_368
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_3_4661dense_3_4663*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_3_layer_call_and_return_conditional_losses_609
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_4_4666dense_4_4668*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_4_layer_call_and_return_conditional_losses_311
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_5_4671dense_5_4673*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_5_layer_call_and_return_conditional_losses_626Ö
lambda/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_lambda_layer_call_and_return_conditional_losses_382ò
add/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0lambda/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *E
f@R>
<__inference_add_layer_call_and_return_conditional_losses_393
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 k
IdentityIdentityadd/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1


ñ
@__inference_dense_2_layer_call_and_return_conditional_losses_368

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


ñ
@__inference_dense_4_layer_call_and_return_conditional_losses_311

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ä
ç
!__inference__traced_restore_11076
file_prefix9
'assignvariableop_dqn_model_dense_kernel:@5
'assignvariableop_1_dqn_model_dense_bias:@=
+assignvariableop_2_dqn_model_dense_1_kernel:@ 7
)assignvariableop_3_dqn_model_dense_1_bias: =
+assignvariableop_4_dqn_model_dense_2_kernel:  7
)assignvariableop_5_dqn_model_dense_2_bias: =
+assignvariableop_6_dqn_model_dense_3_kernel: 7
)assignvariableop_7_dqn_model_dense_3_bias:=
+assignvariableop_8_dqn_model_dense_4_kernel:  7
)assignvariableop_9_dqn_model_dense_4_bias: >
,assignvariableop_10_dqn_model_dense_5_kernel: 8
*assignvariableop_11_dqn_model_dense_5_bias:'
assignvariableop_12_adam_iter:	 )
assignvariableop_13_adam_beta_1: )
assignvariableop_14_adam_beta_2: (
assignvariableop_15_adam_decay: 0
&assignvariableop_16_adam_learning_rate: #
assignvariableop_17_total: #
assignvariableop_18_count: C
1assignvariableop_19_adam_dqn_model_dense_kernel_m:@=
/assignvariableop_20_adam_dqn_model_dense_bias_m:@E
3assignvariableop_21_adam_dqn_model_dense_1_kernel_m:@ ?
1assignvariableop_22_adam_dqn_model_dense_1_bias_m: E
3assignvariableop_23_adam_dqn_model_dense_3_kernel_m: ?
1assignvariableop_24_adam_dqn_model_dense_3_bias_m:E
3assignvariableop_25_adam_dqn_model_dense_5_kernel_m: ?
1assignvariableop_26_adam_dqn_model_dense_5_bias_m:C
1assignvariableop_27_adam_dqn_model_dense_kernel_v:@=
/assignvariableop_28_adam_dqn_model_dense_bias_v:@E
3assignvariableop_29_adam_dqn_model_dense_1_kernel_v:@ ?
1assignvariableop_30_adam_dqn_model_dense_1_bias_v: E
3assignvariableop_31_adam_dqn_model_dense_3_kernel_v: ?
1assignvariableop_32_adam_dqn_model_dense_3_bias_v:E
3assignvariableop_33_adam_dqn_model_dense_5_kernel_v: ?
1assignvariableop_34_adam_dqn_model_dense_5_bias_v:
identity_36¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¢
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*È
value¾B»$B(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense2/bias/.ATTRIBUTES/VARIABLE_VALUEB+adv_dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB)adv_dense/bias/.ATTRIBUTES/VARIABLE_VALUEB)adv_out/kernel/.ATTRIBUTES/VARIABLE_VALUEB'adv_out/bias/.ATTRIBUTES/VARIABLE_VALUEB)v_dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB'v_dense/bias/.ATTRIBUTES/VARIABLE_VALUEB'v_out/kernel/.ATTRIBUTES/VARIABLE_VALUEB%v_out/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBDdense1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBEadv_out/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCadv_out/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCv_out/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAv_out/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDdense2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBEadv_out/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCadv_out/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCv_out/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAv_out/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¸
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Õ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¦
_output_shapes
::::::::::::::::::::::::::::::::::::*2
dtypes(
&2$	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp'assignvariableop_dqn_model_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp'assignvariableop_1_dqn_model_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp+assignvariableop_2_dqn_model_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp)assignvariableop_3_dqn_model_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp+assignvariableop_4_dqn_model_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp)assignvariableop_5_dqn_model_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp+assignvariableop_6_dqn_model_dense_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp)assignvariableop_7_dqn_model_dense_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp+assignvariableop_8_dqn_model_dense_4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp)assignvariableop_9_dqn_model_dense_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp,assignvariableop_10_dqn_model_dense_5_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp*assignvariableop_11_dqn_model_dense_5_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_19AssignVariableOp1assignvariableop_19_adam_dqn_model_dense_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_20AssignVariableOp/assignvariableop_20_adam_dqn_model_dense_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_21AssignVariableOp3assignvariableop_21_adam_dqn_model_dense_1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_22AssignVariableOp1assignvariableop_22_adam_dqn_model_dense_1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_23AssignVariableOp3assignvariableop_23_adam_dqn_model_dense_3_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_24AssignVariableOp1assignvariableop_24_adam_dqn_model_dense_3_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_25AssignVariableOp3assignvariableop_25_adam_dqn_model_dense_5_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_26AssignVariableOp1assignvariableop_26_adam_dqn_model_dense_5_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_27AssignVariableOp1assignvariableop_27_adam_dqn_model_dense_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_28AssignVariableOp/assignvariableop_28_adam_dqn_model_dense_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_29AssignVariableOp3assignvariableop_29_adam_dqn_model_dense_1_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_30AssignVariableOp1assignvariableop_30_adam_dqn_model_dense_1_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_31AssignVariableOp3assignvariableop_31_adam_dqn_model_dense_3_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_32AssignVariableOp1assignvariableop_32_adam_dqn_model_dense_3_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_33AssignVariableOp3assignvariableop_33_adam_dqn_model_dense_5_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_34AssignVariableOp1assignvariableop_34_adam_dqn_model_dense_5_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ñ
Identity_35Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_36IdentityIdentity_35:output:0^NoOp_1*
T0*
_output_shapes
: ¾
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_36Identity_36:output:0*[
_input_shapesJ
H: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
J

__inference__traced_save_10961
file_prefix5
1savev2_dqn_model_dense_kernel_read_readvariableop3
/savev2_dqn_model_dense_bias_read_readvariableop7
3savev2_dqn_model_dense_1_kernel_read_readvariableop5
1savev2_dqn_model_dense_1_bias_read_readvariableop7
3savev2_dqn_model_dense_2_kernel_read_readvariableop5
1savev2_dqn_model_dense_2_bias_read_readvariableop7
3savev2_dqn_model_dense_3_kernel_read_readvariableop5
1savev2_dqn_model_dense_3_bias_read_readvariableop7
3savev2_dqn_model_dense_4_kernel_read_readvariableop5
1savev2_dqn_model_dense_4_bias_read_readvariableop7
3savev2_dqn_model_dense_5_kernel_read_readvariableop5
1savev2_dqn_model_dense_5_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop<
8savev2_adam_dqn_model_dense_kernel_m_read_readvariableop:
6savev2_adam_dqn_model_dense_bias_m_read_readvariableop>
:savev2_adam_dqn_model_dense_1_kernel_m_read_readvariableop<
8savev2_adam_dqn_model_dense_1_bias_m_read_readvariableop>
:savev2_adam_dqn_model_dense_3_kernel_m_read_readvariableop<
8savev2_adam_dqn_model_dense_3_bias_m_read_readvariableop>
:savev2_adam_dqn_model_dense_5_kernel_m_read_readvariableop<
8savev2_adam_dqn_model_dense_5_bias_m_read_readvariableop<
8savev2_adam_dqn_model_dense_kernel_v_read_readvariableop:
6savev2_adam_dqn_model_dense_bias_v_read_readvariableop>
:savev2_adam_dqn_model_dense_1_kernel_v_read_readvariableop<
8savev2_adam_dqn_model_dense_1_bias_v_read_readvariableop>
:savev2_adam_dqn_model_dense_3_kernel_v_read_readvariableop<
8savev2_adam_dqn_model_dense_3_bias_v_read_readvariableop>
:savev2_adam_dqn_model_dense_5_kernel_v_read_readvariableop<
8savev2_adam_dqn_model_dense_5_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*È
value¾B»$B(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense2/bias/.ATTRIBUTES/VARIABLE_VALUEB+adv_dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB)adv_dense/bias/.ATTRIBUTES/VARIABLE_VALUEB)adv_out/kernel/.ATTRIBUTES/VARIABLE_VALUEB'adv_out/bias/.ATTRIBUTES/VARIABLE_VALUEB)v_dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB'v_dense/bias/.ATTRIBUTES/VARIABLE_VALUEB'v_out/kernel/.ATTRIBUTES/VARIABLE_VALUEB%v_out/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBDdense1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBEadv_out/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCadv_out/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCv_out/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAv_out/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDdense2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBEadv_out/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCadv_out/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCv_out/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAv_out/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHµ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ÷
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_dqn_model_dense_kernel_read_readvariableop/savev2_dqn_model_dense_bias_read_readvariableop3savev2_dqn_model_dense_1_kernel_read_readvariableop1savev2_dqn_model_dense_1_bias_read_readvariableop3savev2_dqn_model_dense_2_kernel_read_readvariableop1savev2_dqn_model_dense_2_bias_read_readvariableop3savev2_dqn_model_dense_3_kernel_read_readvariableop1savev2_dqn_model_dense_3_bias_read_readvariableop3savev2_dqn_model_dense_4_kernel_read_readvariableop1savev2_dqn_model_dense_4_bias_read_readvariableop3savev2_dqn_model_dense_5_kernel_read_readvariableop1savev2_dqn_model_dense_5_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop8savev2_adam_dqn_model_dense_kernel_m_read_readvariableop6savev2_adam_dqn_model_dense_bias_m_read_readvariableop:savev2_adam_dqn_model_dense_1_kernel_m_read_readvariableop8savev2_adam_dqn_model_dense_1_bias_m_read_readvariableop:savev2_adam_dqn_model_dense_3_kernel_m_read_readvariableop8savev2_adam_dqn_model_dense_3_bias_m_read_readvariableop:savev2_adam_dqn_model_dense_5_kernel_m_read_readvariableop8savev2_adam_dqn_model_dense_5_bias_m_read_readvariableop8savev2_adam_dqn_model_dense_kernel_v_read_readvariableop6savev2_adam_dqn_model_dense_bias_v_read_readvariableop:savev2_adam_dqn_model_dense_1_kernel_v_read_readvariableop8savev2_adam_dqn_model_dense_1_bias_v_read_readvariableop:savev2_adam_dqn_model_dense_3_kernel_v_read_readvariableop8savev2_adam_dqn_model_dense_3_bias_v_read_readvariableop:savev2_adam_dqn_model_dense_5_kernel_v_read_readvariableop8savev2_adam_dqn_model_dense_5_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *2
dtypes(
&2$	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapesõ
ò: :@:@:@ : :  : : ::  : : :: : : : : : : :@:@:@ : : :: ::@:@:@ : : :: :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$	 

_output_shapes

:  : 


_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$  

_output_shapes

: : !

_output_shapes
::$" 

_output_shapes

: : #

_output_shapes
::$

_output_shapes
: 
ó$

B__inference_dqn_model_layer_call_and_return_conditional_losses_755
input_1

dense_4682:@

dense_4684:@
dense_1_4687:@ 
dense_1_4689: 
dense_2_4692:  
dense_2_4694: 
dense_3_4697: 
dense_3_4699:
dense_4_4702:  
dense_4_4704: 
dense_5_4707: 
dense_5_4709:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCallá
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1
dense_4682
dense_4684*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_350
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_4687dense_1_4689*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_666
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_4692dense_2_4694*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_2_layer_call_and_return_conditional_losses_368
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_3_4697dense_3_4699*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_3_layer_call_and_return_conditional_losses_609
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_4_4702dense_4_4704*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_4_layer_call_and_return_conditional_losses_311
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_5_4707dense_5_4709*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_5_layer_call_and_return_conditional_losses_626Ö
lambda/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_lambda_layer_call_and_return_conditional_losses_417ò
add/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0lambda/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *E
f@R>
<__inference_add_layer_call_and_return_conditional_losses_393
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 k
IdentityIdentityadd/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
5
	
B__inference_dqn_model_layer_call_and_return_conditional_losses_248	
input6
$dense_matmul_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@ 5
'dense_1_biasadd_readvariableop_resource: 8
&dense_2_matmul_readvariableop_resource:  5
'dense_2_biasadd_readvariableop_resource: 8
&dense_3_matmul_readvariableop_resource: 5
'dense_3_biasadd_readvariableop_resource:8
&dense_4_matmul_readvariableop_resource:  5
'dense_4_biasadd_readvariableop_resource: 8
&dense_5_matmul_readvariableop_resource: 5
'dense_5_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢dense_3/BiasAdd/ReadVariableOp¢dense_3/MatMul/ReadVariableOp¢dense_4/BiasAdd/ReadVariableOp¢dense_4/MatMul/ReadVariableOp¢dense_5/BiasAdd/ReadVariableOp¢dense_5/MatMul/ReadVariableOp
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0t
dense/MatMulMatMulinput#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_3/MatMulMatMuldense_1/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_4/MatMulMatMuldense_1/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_5/MatMulMatMuldense_1/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
lambda/ConstConst*
_output_shapes
:*
dtype0*
valueB"       e
lambda/MeanMeandense_3/BiasAdd:output:0lambda/Const:output:0*
T0*
_output_shapes
: s

lambda/subSubdense_3/BiasAdd:output:0lambda/Mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
add/addAddV2dense_5/BiasAdd:output:0lambda/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 Z
IdentityIdentityadd/add:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
5
	
B__inference_dqn_model_layer_call_and_return_conditional_losses_599	
input6
$dense_matmul_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@ 5
'dense_1_biasadd_readvariableop_resource: 8
&dense_2_matmul_readvariableop_resource:  5
'dense_2_biasadd_readvariableop_resource: 8
&dense_3_matmul_readvariableop_resource: 5
'dense_3_biasadd_readvariableop_resource:8
&dense_4_matmul_readvariableop_resource:  5
'dense_4_biasadd_readvariableop_resource: 8
&dense_5_matmul_readvariableop_resource: 5
'dense_5_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢dense_3/BiasAdd/ReadVariableOp¢dense_3/MatMul/ReadVariableOp¢dense_4/BiasAdd/ReadVariableOp¢dense_4/MatMul/ReadVariableOp¢dense_5/BiasAdd/ReadVariableOp¢dense_5/MatMul/ReadVariableOp
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0t
dense/MatMulMatMulinput#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_3/MatMulMatMuldense_1/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0
dense_4/MatMulMatMuldense_1/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_5/MatMulMatMuldense_1/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
lambda/ConstConst*
_output_shapes
:*
dtype0*
valueB"       e
lambda/MeanMeandense_3/BiasAdd:output:0lambda/Const:output:0*
T0*
_output_shapes
: s

lambda/subSubdense_3/BiasAdd:output:0lambda/Mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
add/addAddV2dense_5/BiasAdd:output:0lambda/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 Z
IdentityIdentityadd/add:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
ä

¢
'__inference_dqn_model_layer_call_fn_714	
input
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6:
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10:
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dqn_model_layer_call_and_return_conditional_losses_697`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
ä

¢
'__inference_dqn_model_layer_call_fn_813	
input
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6:
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10:
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dqn_model_layer_call_and_return_conditional_losses_779`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
í$

B__inference_dqn_model_layer_call_and_return_conditional_losses_779	
input

dense_4554:@

dense_4556:@
dense_1_4559:@ 
dense_1_4561: 
dense_2_4564:  
dense_2_4566: 
dense_3_4569: 
dense_3_4571:
dense_4_4574:  
dense_4_4576: 
dense_5_4579: 
dense_5_4581:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCallß
dense/StatefulPartitionedCallStatefulPartitionedCallinput
dense_4554
dense_4556*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_350
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_4559dense_1_4561*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_666
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_4564dense_2_4566*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_2_layer_call_and_return_conditional_losses_368
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_3_4569dense_3_4571*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_3_layer_call_and_return_conditional_losses_609
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_4_4574dense_4_4576*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_4_layer_call_and_return_conditional_losses_311
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_5_4579dense_5_4581*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_5_layer_call_and_return_conditional_losses_626Ö
lambda/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_lambda_layer_call_and_return_conditional_losses_417ò
add/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0lambda/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *E
f@R>
<__inference_add_layer_call_and_return_conditional_losses_393
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 k
IdentityIdentityadd/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
Ã	
ñ
@__inference_dense_3_layer_call_and_return_conditional_losses_609

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Á


#__inference_signature_wrapper_10833

args_0
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6:
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10:
identity¢StatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_10796o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameargs_0"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*©
serving_default
9
args_0/
serving_default_args_0:0ÿÿÿÿÿÿÿÿÿ<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:y
ñ

dense1

dense2
	adv_dense
adv_out
v_dense
	v_out
lambda_layer
combine
		optimizer


signatures
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature"
_tf_keras_model
à

kernel
bias
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
à

kernel
bias
#_self_saveable_object_factories
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses"
_tf_keras_layer
à

%kernel
&bias
#'_self_saveable_object_factories
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses"
_tf_keras_layer
à

.kernel
/bias
#0_self_saveable_object_factories
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses"
_tf_keras_layer
à

7kernel
8bias
#9_self_saveable_object_factories
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses"
_tf_keras_layer
à

@kernel
Abias
#B_self_saveable_object_factories
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses"
_tf_keras_layer
Ê
#I_self_saveable_object_factories
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses"
_tf_keras_layer
Ê
#P_self_saveable_object_factories
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses"
_tf_keras_layer
ó
Witer

Xbeta_1

Ybeta_2
	Zdecay
[learning_ratemmmm.m/m@mAmvvvv.v/v@vAv"
	optimizer
,
\serving_default"
signature_map
 "
trackable_dict_wrapper
v
0
1
2
3
%4
&5
.6
/7
78
89
@10
A11"
trackable_list_wrapper
v
0
1
2
3
%4
&5
.6
/7
78
89
@10
A11"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
'__inference_dqn_model_layer_call_fn_731
'__inference_dqn_model_layer_call_fn_714
'__inference_dqn_model_layer_call_fn_813
'__inference_dqn_model_layer_call_fn_796¨
¡²
FullArgSpec 
args
jinput

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¾2»
B__inference_dqn_model_layer_call_and_return_conditional_losses_599
B__inference_dqn_model_layer_call_and_return_conditional_losses_248
B__inference_dqn_model_layer_call_and_return_conditional_losses_837
B__inference_dqn_model_layer_call_and_return_conditional_losses_755¨
¡²
FullArgSpec 
args
jinput

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÊBÇ
 __inference__wrapped_model_10796args_0"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
(:&@2dqn_model/dense/kernel
": @2dqn_model/dense/bias
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
*:(@ 2dqn_model/dense_1/kernel
$:" 2dqn_model/dense_1/bias
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
*:(  2dqn_model/dense_2/kernel
$:" 2dqn_model/dense_2/bias
 "
trackable_dict_wrapper
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
*:( 2dqn_model/dense_3/kernel
$:"2dqn_model/dense_3/bias
 "
trackable_dict_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
*:(  2dqn_model/dense_4/kernel
$:" 2dqn_model/dense_4/bias
 "
trackable_dict_wrapper
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
­
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
*:( 2dqn_model/dense_5/kernel
$:"2dqn_model/dense_5/bias
 "
trackable_dict_wrapper
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
Æ2ÃÀ
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2ÃÀ
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ÉBÆ
#__inference_signature_wrapper_10833args_0"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

total

count
	variables
	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
-:+@2Adam/dqn_model/dense/kernel/m
':%@2Adam/dqn_model/dense/bias/m
/:-@ 2Adam/dqn_model/dense_1/kernel/m
):' 2Adam/dqn_model/dense_1/bias/m
/:- 2Adam/dqn_model/dense_3/kernel/m
):'2Adam/dqn_model/dense_3/bias/m
/:- 2Adam/dqn_model/dense_5/kernel/m
):'2Adam/dqn_model/dense_5/bias/m
-:+@2Adam/dqn_model/dense/kernel/v
':%@2Adam/dqn_model/dense/bias/v
/:-@ 2Adam/dqn_model/dense_1/kernel/v
):' 2Adam/dqn_model/dense_1/bias/v
/:- 2Adam/dqn_model/dense_3/kernel/v
):'2Adam/dqn_model/dense_3/bias/v
/:- 2Adam/dqn_model/dense_5/kernel/v
):'2Adam/dqn_model/dense_5/bias/v
 __inference__wrapped_model_10796t%&./78@A/¢,
%¢"
 
args_0ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ¯
B__inference_dqn_model_layer_call_and_return_conditional_losses_248i%&./78@A2¢/
(¢%

inputÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¯
B__inference_dqn_model_layer_call_and_return_conditional_losses_599i%&./78@A2¢/
(¢%

inputÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ±
B__inference_dqn_model_layer_call_and_return_conditional_losses_755k%&./78@A4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ±
B__inference_dqn_model_layer_call_and_return_conditional_losses_837k%&./78@A4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
'__inference_dqn_model_layer_call_fn_714\%&./78@A2¢/
(¢%

inputÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
'__inference_dqn_model_layer_call_fn_731^%&./78@A4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
'__inference_dqn_model_layer_call_fn_796^%&./78@A4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
'__inference_dqn_model_layer_call_fn_813\%&./78@A2¢/
(¢%

inputÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¥
#__inference_signature_wrapper_10833~%&./78@A9¢6
¢ 
/ª,
*
args_0 
args_0ÿÿÿÿÿÿÿÿÿ"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ