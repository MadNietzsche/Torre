??.
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
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
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??(
?
sequential_1/dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *-
shared_namesequential_1/dense_14/kernel
?
0sequential_1/dense_14/kernel/Read/ReadVariableOpReadVariableOpsequential_1/dense_14/kernel*
_output_shapes

: *
dtype0
?
sequential_1/dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namesequential_1/dense_14/bias
?
.sequential_1/dense_14/bias/Read/ReadVariableOpReadVariableOpsequential_1/dense_14/bias*
_output_shapes
: *
dtype0
?
)sequential_1/batch_normalization_11/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)sequential_1/batch_normalization_11/gamma
?
=sequential_1/batch_normalization_11/gamma/Read/ReadVariableOpReadVariableOp)sequential_1/batch_normalization_11/gamma*
_output_shapes
: *
dtype0
?
(sequential_1/batch_normalization_11/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(sequential_1/batch_normalization_11/beta
?
<sequential_1/batch_normalization_11/beta/Read/ReadVariableOpReadVariableOp(sequential_1/batch_normalization_11/beta*
_output_shapes
: *
dtype0
?
/sequential_1/batch_normalization_11/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/sequential_1/batch_normalization_11/moving_mean
?
Csequential_1/batch_normalization_11/moving_mean/Read/ReadVariableOpReadVariableOp/sequential_1/batch_normalization_11/moving_mean*
_output_shapes
: *
dtype0
?
3sequential_1/batch_normalization_11/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *D
shared_name53sequential_1/batch_normalization_11/moving_variance
?
Gsequential_1/batch_normalization_11/moving_variance/Read/ReadVariableOpReadVariableOp3sequential_1/batch_normalization_11/moving_variance*
_output_shapes
: *
dtype0
?
sequential_1/dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*-
shared_namesequential_1/dense_13/kernel
?
0sequential_1/dense_13/kernel/Read/ReadVariableOpReadVariableOpsequential_1/dense_13/kernel*
_output_shapes
:	 ?*
dtype0
?
sequential_1/dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_namesequential_1/dense_13/bias
?
.sequential_1/dense_13/bias/Read/ReadVariableOpReadVariableOpsequential_1/dense_13/bias*
_output_shapes	
:?*
dtype0
?
)sequential_1/batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*:
shared_name+)sequential_1/batch_normalization_10/gamma
?
=sequential_1/batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOp)sequential_1/batch_normalization_10/gamma*
_output_shapes	
:?*
dtype0
?
(sequential_1/batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*9
shared_name*(sequential_1/batch_normalization_10/beta
?
<sequential_1/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOp(sequential_1/batch_normalization_10/beta*
_output_shapes	
:?*
dtype0
?
/sequential_1/batch_normalization_10/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*@
shared_name1/sequential_1/batch_normalization_10/moving_mean
?
Csequential_1/batch_normalization_10/moving_mean/Read/ReadVariableOpReadVariableOp/sequential_1/batch_normalization_10/moving_mean*
_output_shapes	
:?*
dtype0
?
3sequential_1/batch_normalization_10/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*D
shared_name53sequential_1/batch_normalization_10/moving_variance
?
Gsequential_1/batch_normalization_10/moving_variance/Read/ReadVariableOpReadVariableOp3sequential_1/batch_normalization_10/moving_variance*
_output_shapes	
:?*
dtype0
?
sequential_1/dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*-
shared_namesequential_1/dense_12/kernel
?
0sequential_1/dense_12/kernel/Read/ReadVariableOpReadVariableOpsequential_1/dense_12/kernel*
_output_shapes
:	?@*
dtype0
?
sequential_1/dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namesequential_1/dense_12/bias
?
.sequential_1/dense_12/bias/Read/ReadVariableOpReadVariableOpsequential_1/dense_12/bias*
_output_shapes
:@*
dtype0
?
(sequential_1/batch_normalization_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(sequential_1/batch_normalization_9/gamma
?
<sequential_1/batch_normalization_9/gamma/Read/ReadVariableOpReadVariableOp(sequential_1/batch_normalization_9/gamma*
_output_shapes
:@*
dtype0
?
'sequential_1/batch_normalization_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'sequential_1/batch_normalization_9/beta
?
;sequential_1/batch_normalization_9/beta/Read/ReadVariableOpReadVariableOp'sequential_1/batch_normalization_9/beta*
_output_shapes
:@*
dtype0
?
.sequential_1/batch_normalization_9/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.sequential_1/batch_normalization_9/moving_mean
?
Bsequential_1/batch_normalization_9/moving_mean/Read/ReadVariableOpReadVariableOp.sequential_1/batch_normalization_9/moving_mean*
_output_shapes
:@*
dtype0
?
2sequential_1/batch_normalization_9/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*C
shared_name42sequential_1/batch_normalization_9/moving_variance
?
Fsequential_1/batch_normalization_9/moving_variance/Read/ReadVariableOpReadVariableOp2sequential_1/batch_normalization_9/moving_variance*
_output_shapes
:@*
dtype0
?
sequential_1/dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*-
shared_namesequential_1/dense_11/kernel
?
0sequential_1/dense_11/kernel/Read/ReadVariableOpReadVariableOpsequential_1/dense_11/kernel*
_output_shapes
:	@?*
dtype0
?
sequential_1/dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_namesequential_1/dense_11/bias
?
.sequential_1/dense_11/bias/Read/ReadVariableOpReadVariableOpsequential_1/dense_11/bias*
_output_shapes	
:?*
dtype0
?
(sequential_1/batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*9
shared_name*(sequential_1/batch_normalization_8/gamma
?
<sequential_1/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOp(sequential_1/batch_normalization_8/gamma*
_output_shapes	
:?*
dtype0
?
'sequential_1/batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*8
shared_name)'sequential_1/batch_normalization_8/beta
?
;sequential_1/batch_normalization_8/beta/Read/ReadVariableOpReadVariableOp'sequential_1/batch_normalization_8/beta*
_output_shapes	
:?*
dtype0
?
.sequential_1/batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.sequential_1/batch_normalization_8/moving_mean
?
Bsequential_1/batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp.sequential_1/batch_normalization_8/moving_mean*
_output_shapes	
:?*
dtype0
?
2sequential_1/batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*C
shared_name42sequential_1/batch_normalization_8/moving_variance
?
Fsequential_1/batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp2sequential_1/batch_normalization_8/moving_variance*
_output_shapes	
:?*
dtype0
?
sequential_1/dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?	*-
shared_namesequential_1/dense_10/kernel
?
0sequential_1/dense_10/kernel/Read/ReadVariableOpReadVariableOpsequential_1/dense_10/kernel*
_output_shapes
:	?	*
dtype0
?
sequential_1/dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*+
shared_namesequential_1/dense_10/bias
?
.sequential_1/dense_10/bias/Read/ReadVariableOpReadVariableOpsequential_1/dense_10/bias*
_output_shapes
:	*
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
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
#Adam/sequential_1/dense_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *4
shared_name%#Adam/sequential_1/dense_14/kernel/m
?
7Adam/sequential_1/dense_14/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/sequential_1/dense_14/kernel/m*
_output_shapes

: *
dtype0
?
!Adam/sequential_1/dense_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/sequential_1/dense_14/bias/m
?
5Adam/sequential_1/dense_14/bias/m/Read/ReadVariableOpReadVariableOp!Adam/sequential_1/dense_14/bias/m*
_output_shapes
: *
dtype0
?
0Adam/sequential_1/batch_normalization_11/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20Adam/sequential_1/batch_normalization_11/gamma/m
?
DAdam/sequential_1/batch_normalization_11/gamma/m/Read/ReadVariableOpReadVariableOp0Adam/sequential_1/batch_normalization_11/gamma/m*
_output_shapes
: *
dtype0
?
/Adam/sequential_1/batch_normalization_11/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/Adam/sequential_1/batch_normalization_11/beta/m
?
CAdam/sequential_1/batch_normalization_11/beta/m/Read/ReadVariableOpReadVariableOp/Adam/sequential_1/batch_normalization_11/beta/m*
_output_shapes
: *
dtype0
?
#Adam/sequential_1/dense_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*4
shared_name%#Adam/sequential_1/dense_13/kernel/m
?
7Adam/sequential_1/dense_13/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/sequential_1/dense_13/kernel/m*
_output_shapes
:	 ?*
dtype0
?
!Adam/sequential_1/dense_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adam/sequential_1/dense_13/bias/m
?
5Adam/sequential_1/dense_13/bias/m/Read/ReadVariableOpReadVariableOp!Adam/sequential_1/dense_13/bias/m*
_output_shapes	
:?*
dtype0
?
0Adam/sequential_1/batch_normalization_10/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*A
shared_name20Adam/sequential_1/batch_normalization_10/gamma/m
?
DAdam/sequential_1/batch_normalization_10/gamma/m/Read/ReadVariableOpReadVariableOp0Adam/sequential_1/batch_normalization_10/gamma/m*
_output_shapes	
:?*
dtype0
?
/Adam/sequential_1/batch_normalization_10/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*@
shared_name1/Adam/sequential_1/batch_normalization_10/beta/m
?
CAdam/sequential_1/batch_normalization_10/beta/m/Read/ReadVariableOpReadVariableOp/Adam/sequential_1/batch_normalization_10/beta/m*
_output_shapes	
:?*
dtype0
?
#Adam/sequential_1/dense_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*4
shared_name%#Adam/sequential_1/dense_12/kernel/m
?
7Adam/sequential_1/dense_12/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/sequential_1/dense_12/kernel/m*
_output_shapes
:	?@*
dtype0
?
!Adam/sequential_1/dense_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/sequential_1/dense_12/bias/m
?
5Adam/sequential_1/dense_12/bias/m/Read/ReadVariableOpReadVariableOp!Adam/sequential_1/dense_12/bias/m*
_output_shapes
:@*
dtype0
?
/Adam/sequential_1/batch_normalization_9/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*@
shared_name1/Adam/sequential_1/batch_normalization_9/gamma/m
?
CAdam/sequential_1/batch_normalization_9/gamma/m/Read/ReadVariableOpReadVariableOp/Adam/sequential_1/batch_normalization_9/gamma/m*
_output_shapes
:@*
dtype0
?
.Adam/sequential_1/batch_normalization_9/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.Adam/sequential_1/batch_normalization_9/beta/m
?
BAdam/sequential_1/batch_normalization_9/beta/m/Read/ReadVariableOpReadVariableOp.Adam/sequential_1/batch_normalization_9/beta/m*
_output_shapes
:@*
dtype0
?
#Adam/sequential_1/dense_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*4
shared_name%#Adam/sequential_1/dense_11/kernel/m
?
7Adam/sequential_1/dense_11/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/sequential_1/dense_11/kernel/m*
_output_shapes
:	@?*
dtype0
?
!Adam/sequential_1/dense_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adam/sequential_1/dense_11/bias/m
?
5Adam/sequential_1/dense_11/bias/m/Read/ReadVariableOpReadVariableOp!Adam/sequential_1/dense_11/bias/m*
_output_shapes	
:?*
dtype0
?
/Adam/sequential_1/batch_normalization_8/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*@
shared_name1/Adam/sequential_1/batch_normalization_8/gamma/m
?
CAdam/sequential_1/batch_normalization_8/gamma/m/Read/ReadVariableOpReadVariableOp/Adam/sequential_1/batch_normalization_8/gamma/m*
_output_shapes	
:?*
dtype0
?
.Adam/sequential_1/batch_normalization_8/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.Adam/sequential_1/batch_normalization_8/beta/m
?
BAdam/sequential_1/batch_normalization_8/beta/m/Read/ReadVariableOpReadVariableOp.Adam/sequential_1/batch_normalization_8/beta/m*
_output_shapes	
:?*
dtype0
?
#Adam/sequential_1/dense_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?	*4
shared_name%#Adam/sequential_1/dense_10/kernel/m
?
7Adam/sequential_1/dense_10/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/sequential_1/dense_10/kernel/m*
_output_shapes
:	?	*
dtype0
?
!Adam/sequential_1/dense_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*2
shared_name#!Adam/sequential_1/dense_10/bias/m
?
5Adam/sequential_1/dense_10/bias/m/Read/ReadVariableOpReadVariableOp!Adam/sequential_1/dense_10/bias/m*
_output_shapes
:	*
dtype0
?
#Adam/sequential_1/dense_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *4
shared_name%#Adam/sequential_1/dense_14/kernel/v
?
7Adam/sequential_1/dense_14/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/sequential_1/dense_14/kernel/v*
_output_shapes

: *
dtype0
?
!Adam/sequential_1/dense_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/sequential_1/dense_14/bias/v
?
5Adam/sequential_1/dense_14/bias/v/Read/ReadVariableOpReadVariableOp!Adam/sequential_1/dense_14/bias/v*
_output_shapes
: *
dtype0
?
0Adam/sequential_1/batch_normalization_11/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20Adam/sequential_1/batch_normalization_11/gamma/v
?
DAdam/sequential_1/batch_normalization_11/gamma/v/Read/ReadVariableOpReadVariableOp0Adam/sequential_1/batch_normalization_11/gamma/v*
_output_shapes
: *
dtype0
?
/Adam/sequential_1/batch_normalization_11/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/Adam/sequential_1/batch_normalization_11/beta/v
?
CAdam/sequential_1/batch_normalization_11/beta/v/Read/ReadVariableOpReadVariableOp/Adam/sequential_1/batch_normalization_11/beta/v*
_output_shapes
: *
dtype0
?
#Adam/sequential_1/dense_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*4
shared_name%#Adam/sequential_1/dense_13/kernel/v
?
7Adam/sequential_1/dense_13/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/sequential_1/dense_13/kernel/v*
_output_shapes
:	 ?*
dtype0
?
!Adam/sequential_1/dense_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adam/sequential_1/dense_13/bias/v
?
5Adam/sequential_1/dense_13/bias/v/Read/ReadVariableOpReadVariableOp!Adam/sequential_1/dense_13/bias/v*
_output_shapes	
:?*
dtype0
?
0Adam/sequential_1/batch_normalization_10/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*A
shared_name20Adam/sequential_1/batch_normalization_10/gamma/v
?
DAdam/sequential_1/batch_normalization_10/gamma/v/Read/ReadVariableOpReadVariableOp0Adam/sequential_1/batch_normalization_10/gamma/v*
_output_shapes	
:?*
dtype0
?
/Adam/sequential_1/batch_normalization_10/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*@
shared_name1/Adam/sequential_1/batch_normalization_10/beta/v
?
CAdam/sequential_1/batch_normalization_10/beta/v/Read/ReadVariableOpReadVariableOp/Adam/sequential_1/batch_normalization_10/beta/v*
_output_shapes	
:?*
dtype0
?
#Adam/sequential_1/dense_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*4
shared_name%#Adam/sequential_1/dense_12/kernel/v
?
7Adam/sequential_1/dense_12/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/sequential_1/dense_12/kernel/v*
_output_shapes
:	?@*
dtype0
?
!Adam/sequential_1/dense_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/sequential_1/dense_12/bias/v
?
5Adam/sequential_1/dense_12/bias/v/Read/ReadVariableOpReadVariableOp!Adam/sequential_1/dense_12/bias/v*
_output_shapes
:@*
dtype0
?
/Adam/sequential_1/batch_normalization_9/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*@
shared_name1/Adam/sequential_1/batch_normalization_9/gamma/v
?
CAdam/sequential_1/batch_normalization_9/gamma/v/Read/ReadVariableOpReadVariableOp/Adam/sequential_1/batch_normalization_9/gamma/v*
_output_shapes
:@*
dtype0
?
.Adam/sequential_1/batch_normalization_9/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.Adam/sequential_1/batch_normalization_9/beta/v
?
BAdam/sequential_1/batch_normalization_9/beta/v/Read/ReadVariableOpReadVariableOp.Adam/sequential_1/batch_normalization_9/beta/v*
_output_shapes
:@*
dtype0
?
#Adam/sequential_1/dense_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*4
shared_name%#Adam/sequential_1/dense_11/kernel/v
?
7Adam/sequential_1/dense_11/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/sequential_1/dense_11/kernel/v*
_output_shapes
:	@?*
dtype0
?
!Adam/sequential_1/dense_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!Adam/sequential_1/dense_11/bias/v
?
5Adam/sequential_1/dense_11/bias/v/Read/ReadVariableOpReadVariableOp!Adam/sequential_1/dense_11/bias/v*
_output_shapes	
:?*
dtype0
?
/Adam/sequential_1/batch_normalization_8/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*@
shared_name1/Adam/sequential_1/batch_normalization_8/gamma/v
?
CAdam/sequential_1/batch_normalization_8/gamma/v/Read/ReadVariableOpReadVariableOp/Adam/sequential_1/batch_normalization_8/gamma/v*
_output_shapes	
:?*
dtype0
?
.Adam/sequential_1/batch_normalization_8/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.Adam/sequential_1/batch_normalization_8/beta/v
?
BAdam/sequential_1/batch_normalization_8/beta/v/Read/ReadVariableOpReadVariableOp.Adam/sequential_1/batch_normalization_8/beta/v*
_output_shapes	
:?*
dtype0
?
#Adam/sequential_1/dense_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?	*4
shared_name%#Adam/sequential_1/dense_10/kernel/v
?
7Adam/sequential_1/dense_10/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/sequential_1/dense_10/kernel/v*
_output_shapes
:	?	*
dtype0
?
!Adam/sequential_1/dense_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*2
shared_name#!Adam/sequential_1/dense_10/bias/v
?
5Adam/sequential_1/dense_10/bias/v/Read/ReadVariableOpReadVariableOp!Adam/sequential_1/dense_10/bias/v*
_output_shapes
:	*
dtype0

NoOpNoOp
?q
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?q
value?pB?p B?p
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
	optimizer
_build_input_shape
	variables
regularization_losses
trainable_variables
	keras_api

signatures
x
_feature_columns

_resources
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
?
axis
	gamma
 beta
!moving_mean
"moving_variance
#	variables
$regularization_losses
%trainable_variables
&	keras_api
h

'kernel
(bias
)	variables
*regularization_losses
+trainable_variables
,	keras_api
?
-axis
	.gamma
/beta
0moving_mean
1moving_variance
2	variables
3regularization_losses
4trainable_variables
5	keras_api
h

6kernel
7bias
8	variables
9regularization_losses
:trainable_variables
;	keras_api
?
<axis
	=gamma
>beta
?moving_mean
@moving_variance
A	variables
Bregularization_losses
Ctrainable_variables
D	keras_api
h

Ekernel
Fbias
G	variables
Hregularization_losses
Itrainable_variables
J	keras_api
?
Kaxis
	Lgamma
Mbeta
Nmoving_mean
Omoving_variance
P	variables
Qregularization_losses
Rtrainable_variables
S	keras_api
h

Tkernel
Ubias
V	variables
Wregularization_losses
Xtrainable_variables
Y	keras_api
?
Ziter

[beta_1

\beta_2
	]decay
^learning_ratem?m?m? m?'m?(m?.m?/m?6m?7m?=m?>m?Em?Fm?Lm?Mm?Tm?Um?v?v?v? v?'v?(v?.v?/v?6v?7v?=v?>v?Ev?Fv?Lv?Mv?Tv?Uv?
 
?
0
1
2
 3
!4
"5
'6
(7
.8
/9
010
111
612
713
=14
>15
?16
@17
E18
F19
L20
M21
N22
O23
T24
U25
 
?
0
1
2
 3
'4
(5
.6
/7
68
79
=10
>11
E12
F13
L14
M15
T16
U17
?
	variables
_metrics
`non_trainable_variables
regularization_losses

alayers
blayer_regularization_losses
clayer_metrics
trainable_variables
 
 
 
 
 
 
?
	variables
dmetrics
enon_trainable_variables
regularization_losses

flayers
glayer_regularization_losses
hlayer_metrics
trainable_variables
hf
VARIABLE_VALUEsequential_1/dense_14/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEsequential_1/dense_14/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
	variables
imetrics
jnon_trainable_variables
regularization_losses

klayers
llayer_regularization_losses
mlayer_metrics
trainable_variables
 
tr
VARIABLE_VALUE)sequential_1/batch_normalization_11/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE(sequential_1/batch_normalization_11/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE/sequential_1/batch_normalization_11/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE3sequential_1/batch_normalization_11/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
 1
!2
"3
 

0
 1
?
#	variables
nmetrics
onon_trainable_variables
$regularization_losses

players
qlayer_regularization_losses
rlayer_metrics
%trainable_variables
hf
VARIABLE_VALUEsequential_1/dense_13/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEsequential_1/dense_13/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1
 

'0
(1
?
)	variables
smetrics
tnon_trainable_variables
*regularization_losses

ulayers
vlayer_regularization_losses
wlayer_metrics
+trainable_variables
 
tr
VARIABLE_VALUE)sequential_1/batch_normalization_10/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE(sequential_1/batch_normalization_10/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE/sequential_1/batch_normalization_10/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE3sequential_1/batch_normalization_10/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

.0
/1
02
13
 

.0
/1
?
2	variables
xmetrics
ynon_trainable_variables
3regularization_losses

zlayers
{layer_regularization_losses
|layer_metrics
4trainable_variables
hf
VARIABLE_VALUEsequential_1/dense_12/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEsequential_1/dense_12/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

60
71
 

60
71
?
8	variables
}metrics
~non_trainable_variables
9regularization_losses

layers
 ?layer_regularization_losses
?layer_metrics
:trainable_variables
 
sq
VARIABLE_VALUE(sequential_1/batch_normalization_9/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE'sequential_1/batch_normalization_9/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE.sequential_1/batch_normalization_9/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2sequential_1/batch_normalization_9/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

=0
>1
?2
@3
 

=0
>1
?
A	variables
?metrics
?non_trainable_variables
Bregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
Ctrainable_variables
hf
VARIABLE_VALUEsequential_1/dense_11/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEsequential_1/dense_11/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

E0
F1
 

E0
F1
?
G	variables
?metrics
?non_trainable_variables
Hregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
Itrainable_variables
 
sq
VARIABLE_VALUE(sequential_1/batch_normalization_8/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE'sequential_1/batch_normalization_8/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE.sequential_1/batch_normalization_8/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2sequential_1/batch_normalization_8/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

L0
M1
N2
O3
 

L0
M1
?
P	variables
?metrics
?non_trainable_variables
Qregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
Rtrainable_variables
hf
VARIABLE_VALUEsequential_1/dense_10/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEsequential_1/dense_10/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

T0
U1
 

T0
U1
?
V	variables
?metrics
?non_trainable_variables
Wregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
Xtrainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
8
!0
"1
02
13
?4
@5
N6
O7
F
0
1
2
3
4
5
6
7
	8

9
 
 
 
 
 
 
 
 
 
 
 
 
 

!0
"1
 
 
 
 
 
 
 
 
 

00
11
 
 
 
 
 
 
 
 
 

?0
@1
 
 
 
 
 
 
 
 
 

N0
O1
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
??
VARIABLE_VALUE#Adam/sequential_1/dense_14/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/sequential_1/dense_14/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/sequential_1/batch_normalization_11/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE/Adam/sequential_1/batch_normalization_11/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/sequential_1/dense_13/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/sequential_1/dense_13/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/sequential_1/batch_normalization_10/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE/Adam/sequential_1/batch_normalization_10/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/sequential_1/dense_12/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/sequential_1/dense_12/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE/Adam/sequential_1/batch_normalization_9/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/sequential_1/batch_normalization_9/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/sequential_1/dense_11/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/sequential_1/dense_11/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE/Adam/sequential_1/batch_normalization_8/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/sequential_1/batch_normalization_8/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/sequential_1/dense_10/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/sequential_1/dense_10/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/sequential_1/dense_14/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/sequential_1/dense_14/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/sequential_1/batch_normalization_11/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE/Adam/sequential_1/batch_normalization_11/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/sequential_1/dense_13/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/sequential_1/dense_13/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/sequential_1/batch_normalization_10/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE/Adam/sequential_1/batch_normalization_10/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/sequential_1/dense_12/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/sequential_1/dense_12/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE/Adam/sequential_1/batch_normalization_9/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/sequential_1/batch_normalization_9/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/sequential_1/dense_11/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/sequential_1/dense_11/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE/Adam/sequential_1/batch_normalization_8/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/sequential_1/batch_normalization_8/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/sequential_1/dense_10/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/sequential_1/dense_10/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
 serving_default_category_embed_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
 serving_default_category_embed_2Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
 serving_default_category_embed_3Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
 serving_default_category_embed_4Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
 serving_default_category_embed_5Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????

serving_default_city_embed_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????

serving_default_city_embed_2Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????

serving_default_city_embed_3Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????

serving_default_city_embed_4Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????

serving_default_city_embed_5Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
x
serving_default_colonPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
y
serving_default_commasPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
w
serving_default_dashPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
y
serving_default_exclamPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
x
serving_default_moneyPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
x
serving_default_monthPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
~
serving_default_parenthesisPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
serving_default_state_embed_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
serving_default_state_embed_2Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
serving_default_state_embed_3Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
serving_default_state_embed_4Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
serving_default_state_embed_5Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
z
serving_default_weekdayPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCall serving_default_category_embed_1 serving_default_category_embed_2 serving_default_category_embed_3 serving_default_category_embed_4 serving_default_category_embed_5serving_default_city_embed_1serving_default_city_embed_2serving_default_city_embed_3serving_default_city_embed_4serving_default_city_embed_5serving_default_colonserving_default_commasserving_default_dashserving_default_exclamserving_default_moneyserving_default_monthserving_default_parenthesisserving_default_state_embed_1serving_default_state_embed_2serving_default_state_embed_3serving_default_state_embed_4serving_default_state_embed_5serving_default_weekdaysequential_1/dense_14/kernelsequential_1/dense_14/bias/sequential_1/batch_normalization_11/moving_mean3sequential_1/batch_normalization_11/moving_variance(sequential_1/batch_normalization_11/beta)sequential_1/batch_normalization_11/gammasequential_1/dense_13/kernelsequential_1/dense_13/bias/sequential_1/batch_normalization_10/moving_mean3sequential_1/batch_normalization_10/moving_variance(sequential_1/batch_normalization_10/beta)sequential_1/batch_normalization_10/gammasequential_1/dense_12/kernelsequential_1/dense_12/bias.sequential_1/batch_normalization_9/moving_mean2sequential_1/batch_normalization_9/moving_variance'sequential_1/batch_normalization_9/beta(sequential_1/batch_normalization_9/gammasequential_1/dense_11/kernelsequential_1/dense_11/bias.sequential_1/batch_normalization_8/moving_mean2sequential_1/batch_normalization_8/moving_variance'sequential_1/batch_normalization_8/beta(sequential_1/batch_normalization_8/gammasequential_1/dense_10/kernelsequential_1/dense_10/bias*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*<
_read_only_resource_inputs
 !"#$%&'()*+,-./0*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_562526
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?"
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename0sequential_1/dense_14/kernel/Read/ReadVariableOp.sequential_1/dense_14/bias/Read/ReadVariableOp=sequential_1/batch_normalization_11/gamma/Read/ReadVariableOp<sequential_1/batch_normalization_11/beta/Read/ReadVariableOpCsequential_1/batch_normalization_11/moving_mean/Read/ReadVariableOpGsequential_1/batch_normalization_11/moving_variance/Read/ReadVariableOp0sequential_1/dense_13/kernel/Read/ReadVariableOp.sequential_1/dense_13/bias/Read/ReadVariableOp=sequential_1/batch_normalization_10/gamma/Read/ReadVariableOp<sequential_1/batch_normalization_10/beta/Read/ReadVariableOpCsequential_1/batch_normalization_10/moving_mean/Read/ReadVariableOpGsequential_1/batch_normalization_10/moving_variance/Read/ReadVariableOp0sequential_1/dense_12/kernel/Read/ReadVariableOp.sequential_1/dense_12/bias/Read/ReadVariableOp<sequential_1/batch_normalization_9/gamma/Read/ReadVariableOp;sequential_1/batch_normalization_9/beta/Read/ReadVariableOpBsequential_1/batch_normalization_9/moving_mean/Read/ReadVariableOpFsequential_1/batch_normalization_9/moving_variance/Read/ReadVariableOp0sequential_1/dense_11/kernel/Read/ReadVariableOp.sequential_1/dense_11/bias/Read/ReadVariableOp<sequential_1/batch_normalization_8/gamma/Read/ReadVariableOp;sequential_1/batch_normalization_8/beta/Read/ReadVariableOpBsequential_1/batch_normalization_8/moving_mean/Read/ReadVariableOpFsequential_1/batch_normalization_8/moving_variance/Read/ReadVariableOp0sequential_1/dense_10/kernel/Read/ReadVariableOp.sequential_1/dense_10/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp7Adam/sequential_1/dense_14/kernel/m/Read/ReadVariableOp5Adam/sequential_1/dense_14/bias/m/Read/ReadVariableOpDAdam/sequential_1/batch_normalization_11/gamma/m/Read/ReadVariableOpCAdam/sequential_1/batch_normalization_11/beta/m/Read/ReadVariableOp7Adam/sequential_1/dense_13/kernel/m/Read/ReadVariableOp5Adam/sequential_1/dense_13/bias/m/Read/ReadVariableOpDAdam/sequential_1/batch_normalization_10/gamma/m/Read/ReadVariableOpCAdam/sequential_1/batch_normalization_10/beta/m/Read/ReadVariableOp7Adam/sequential_1/dense_12/kernel/m/Read/ReadVariableOp5Adam/sequential_1/dense_12/bias/m/Read/ReadVariableOpCAdam/sequential_1/batch_normalization_9/gamma/m/Read/ReadVariableOpBAdam/sequential_1/batch_normalization_9/beta/m/Read/ReadVariableOp7Adam/sequential_1/dense_11/kernel/m/Read/ReadVariableOp5Adam/sequential_1/dense_11/bias/m/Read/ReadVariableOpCAdam/sequential_1/batch_normalization_8/gamma/m/Read/ReadVariableOpBAdam/sequential_1/batch_normalization_8/beta/m/Read/ReadVariableOp7Adam/sequential_1/dense_10/kernel/m/Read/ReadVariableOp5Adam/sequential_1/dense_10/bias/m/Read/ReadVariableOp7Adam/sequential_1/dense_14/kernel/v/Read/ReadVariableOp5Adam/sequential_1/dense_14/bias/v/Read/ReadVariableOpDAdam/sequential_1/batch_normalization_11/gamma/v/Read/ReadVariableOpCAdam/sequential_1/batch_normalization_11/beta/v/Read/ReadVariableOp7Adam/sequential_1/dense_13/kernel/v/Read/ReadVariableOp5Adam/sequential_1/dense_13/bias/v/Read/ReadVariableOpDAdam/sequential_1/batch_normalization_10/gamma/v/Read/ReadVariableOpCAdam/sequential_1/batch_normalization_10/beta/v/Read/ReadVariableOp7Adam/sequential_1/dense_12/kernel/v/Read/ReadVariableOp5Adam/sequential_1/dense_12/bias/v/Read/ReadVariableOpCAdam/sequential_1/batch_normalization_9/gamma/v/Read/ReadVariableOpBAdam/sequential_1/batch_normalization_9/beta/v/Read/ReadVariableOp7Adam/sequential_1/dense_11/kernel/v/Read/ReadVariableOp5Adam/sequential_1/dense_11/bias/v/Read/ReadVariableOpCAdam/sequential_1/batch_normalization_8/gamma/v/Read/ReadVariableOpBAdam/sequential_1/batch_normalization_8/beta/v/Read/ReadVariableOp7Adam/sequential_1/dense_10/kernel/v/Read/ReadVariableOp5Adam/sequential_1/dense_10/bias/v/Read/ReadVariableOpConst*T
TinM
K2I	*
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
GPU2*0J 8? *(
f#R!
__inference__traced_save_564588
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamesequential_1/dense_14/kernelsequential_1/dense_14/bias)sequential_1/batch_normalization_11/gamma(sequential_1/batch_normalization_11/beta/sequential_1/batch_normalization_11/moving_mean3sequential_1/batch_normalization_11/moving_variancesequential_1/dense_13/kernelsequential_1/dense_13/bias)sequential_1/batch_normalization_10/gamma(sequential_1/batch_normalization_10/beta/sequential_1/batch_normalization_10/moving_mean3sequential_1/batch_normalization_10/moving_variancesequential_1/dense_12/kernelsequential_1/dense_12/bias(sequential_1/batch_normalization_9/gamma'sequential_1/batch_normalization_9/beta.sequential_1/batch_normalization_9/moving_mean2sequential_1/batch_normalization_9/moving_variancesequential_1/dense_11/kernelsequential_1/dense_11/bias(sequential_1/batch_normalization_8/gamma'sequential_1/batch_normalization_8/beta.sequential_1/batch_normalization_8/moving_mean2sequential_1/batch_normalization_8/moving_variancesequential_1/dense_10/kernelsequential_1/dense_10/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1#Adam/sequential_1/dense_14/kernel/m!Adam/sequential_1/dense_14/bias/m0Adam/sequential_1/batch_normalization_11/gamma/m/Adam/sequential_1/batch_normalization_11/beta/m#Adam/sequential_1/dense_13/kernel/m!Adam/sequential_1/dense_13/bias/m0Adam/sequential_1/batch_normalization_10/gamma/m/Adam/sequential_1/batch_normalization_10/beta/m#Adam/sequential_1/dense_12/kernel/m!Adam/sequential_1/dense_12/bias/m/Adam/sequential_1/batch_normalization_9/gamma/m.Adam/sequential_1/batch_normalization_9/beta/m#Adam/sequential_1/dense_11/kernel/m!Adam/sequential_1/dense_11/bias/m/Adam/sequential_1/batch_normalization_8/gamma/m.Adam/sequential_1/batch_normalization_8/beta/m#Adam/sequential_1/dense_10/kernel/m!Adam/sequential_1/dense_10/bias/m#Adam/sequential_1/dense_14/kernel/v!Adam/sequential_1/dense_14/bias/v0Adam/sequential_1/batch_normalization_11/gamma/v/Adam/sequential_1/batch_normalization_11/beta/v#Adam/sequential_1/dense_13/kernel/v!Adam/sequential_1/dense_13/bias/v0Adam/sequential_1/batch_normalization_10/gamma/v/Adam/sequential_1/batch_normalization_10/beta/v#Adam/sequential_1/dense_12/kernel/v!Adam/sequential_1/dense_12/bias/v/Adam/sequential_1/batch_normalization_9/gamma/v.Adam/sequential_1/batch_normalization_9/beta/v#Adam/sequential_1/dense_11/kernel/v!Adam/sequential_1/dense_11/bias/v/Adam/sequential_1/batch_normalization_8/gamma/v.Adam/sequential_1/batch_normalization_8/beta/v#Adam/sequential_1/dense_10/kernel/v!Adam/sequential_1/dense_10/bias/v*S
TinL
J2H*
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
GPU2*0J 8? *+
f&R$
"__inference__traced_restore_564811??%
?
?
6__inference_batch_normalization_9_layer_call_fn_564134

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_5608672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?k
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_562093

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13
	inputs_14
	inputs_15
	inputs_16
	inputs_17
	inputs_18
	inputs_19
	inputs_20
	inputs_21
	inputs_22!
dense_14_562019: 
dense_14_562021: +
batch_normalization_11_562024: +
batch_normalization_11_562026: +
batch_normalization_11_562028: +
batch_normalization_11_562030: "
dense_13_562033:	 ?
dense_13_562035:	?,
batch_normalization_10_562038:	?,
batch_normalization_10_562040:	?,
batch_normalization_10_562042:	?,
batch_normalization_10_562044:	?"
dense_12_562047:	?@
dense_12_562049:@*
batch_normalization_9_562052:@*
batch_normalization_9_562054:@*
batch_normalization_9_562056:@*
batch_normalization_9_562058:@"
dense_11_562061:	@?
dense_11_562063:	?+
batch_normalization_8_562066:	?+
batch_normalization_8_562068:	?+
batch_normalization_8_562070:	?+
batch_normalization_8_562072:	?"
dense_10_562075:	?	
dense_10_562077:	
identity??.batch_normalization_10/StatefulPartitionedCall?.batch_normalization_11/StatefulPartitionedCall?-batch_normalization_8/StatefulPartitionedCall?-batch_normalization_9/StatefulPartitionedCall? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall? dense_14/StatefulPartitionedCall?>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp?>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp?
 dense_features_1/PartitionedCallPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12	inputs_13	inputs_14	inputs_15	inputs_16	inputs_17	inputs_18	inputs_19	inputs_20	inputs_21	inputs_22*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_dense_features_1_layer_call_and_return_conditional_losses_5618872"
 dense_features_1/PartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_features_1/PartitionedCall:output:0dense_14_562019dense_14_562021*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_5613832"
 dense_14/StatefulPartitionedCall?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0batch_normalization_11_562024batch_normalization_11_562026batch_normalization_11_562028batch_normalization_11_562030*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_56054320
.batch_normalization_11/StatefulPartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0dense_13_562033dense_13_562035*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_5614152"
 dense_13/StatefulPartitionedCall?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0batch_normalization_10_562038batch_normalization_10_562040batch_normalization_10_562042batch_normalization_10_562044*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_56070520
.batch_normalization_10/StatefulPartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0dense_12_562047dense_12_562049*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_5614472"
 dense_12/StatefulPartitionedCall?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0batch_normalization_9_562052batch_normalization_9_562054batch_normalization_9_562056batch_normalization_9_562058*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_5608672/
-batch_normalization_9/StatefulPartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0dense_11_562061dense_11_562063*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_5614732"
 dense_11/StatefulPartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0batch_normalization_8_562066batch_normalization_8_562068batch_normalization_8_562070batch_normalization_8_562072*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_5610292/
-batch_normalization_8/StatefulPartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0dense_10_562075dense_10_562077*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_5614992"
 dense_10/StatefulPartitionedCall?
>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_13_562033*
_output_shapes
:	 ?*
dtype02@
>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp?
/sequential_1/dense_13/kernel/Regularizer/SquareSquareFsequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 ?21
/sequential_1/dense_13/kernel/Regularizer/Square?
.sequential_1/dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       20
.sequential_1/dense_13/kernel/Regularizer/Const?
,sequential_1/dense_13/kernel/Regularizer/SumSum3sequential_1/dense_13/kernel/Regularizer/Square:y:07sequential_1/dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2.
,sequential_1/dense_13/kernel/Regularizer/Sum?
.sequential_1/dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_1/dense_13/kernel/Regularizer/mul/x?
,sequential_1/dense_13/kernel/Regularizer/mulMul7sequential_1/dense_13/kernel/Regularizer/mul/x:output:05sequential_1/dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,sequential_1/dense_13/kernel/Regularizer/mul?
>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_12_562047*
_output_shapes
:	?@*
dtype02@
>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp?
/sequential_1/dense_12/kernel/Regularizer/SquareSquareFsequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?@21
/sequential_1/dense_12/kernel/Regularizer/Square?
.sequential_1/dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       20
.sequential_1/dense_12/kernel/Regularizer/Const?
,sequential_1/dense_12/kernel/Regularizer/SumSum3sequential_1/dense_12/kernel/Regularizer/Square:y:07sequential_1/dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2.
,sequential_1/dense_12/kernel/Regularizer/Sum?
.sequential_1/dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>20
.sequential_1/dense_12/kernel/Regularizer/mul/x?
,sequential_1/dense_12/kernel/Regularizer/mulMul7sequential_1/dense_12/kernel/Regularizer/mul/x:output:05sequential_1/dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,sequential_1/dense_12/kernel/Regularizer/mul?
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall?^sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp?^sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2?
>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp2?
>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O	K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O
K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
!__inference__wrapped_model_560459
category_embed_1
category_embed_2
category_embed_3
category_embed_4
category_embed_5
city_embed_1
city_embed_2
city_embed_3
city_embed_4
city_embed_5	
colon

commas
dash

exclam	
money	
month
parenthesis
state_embed_1
state_embed_2
state_embed_3
state_embed_4
state_embed_5
weekdayF
4sequential_1_dense_14_matmul_readvariableop_resource: C
5sequential_1_dense_14_biasadd_readvariableop_resource: N
@sequential_1_batch_normalization_11_cast_readvariableop_resource: P
Bsequential_1_batch_normalization_11_cast_1_readvariableop_resource: P
Bsequential_1_batch_normalization_11_cast_2_readvariableop_resource: P
Bsequential_1_batch_normalization_11_cast_3_readvariableop_resource: G
4sequential_1_dense_13_matmul_readvariableop_resource:	 ?D
5sequential_1_dense_13_biasadd_readvariableop_resource:	?O
@sequential_1_batch_normalization_10_cast_readvariableop_resource:	?Q
Bsequential_1_batch_normalization_10_cast_1_readvariableop_resource:	?Q
Bsequential_1_batch_normalization_10_cast_2_readvariableop_resource:	?Q
Bsequential_1_batch_normalization_10_cast_3_readvariableop_resource:	?G
4sequential_1_dense_12_matmul_readvariableop_resource:	?@C
5sequential_1_dense_12_biasadd_readvariableop_resource:@M
?sequential_1_batch_normalization_9_cast_readvariableop_resource:@O
Asequential_1_batch_normalization_9_cast_1_readvariableop_resource:@O
Asequential_1_batch_normalization_9_cast_2_readvariableop_resource:@O
Asequential_1_batch_normalization_9_cast_3_readvariableop_resource:@G
4sequential_1_dense_11_matmul_readvariableop_resource:	@?D
5sequential_1_dense_11_biasadd_readvariableop_resource:	?N
?sequential_1_batch_normalization_8_cast_readvariableop_resource:	?P
Asequential_1_batch_normalization_8_cast_1_readvariableop_resource:	?P
Asequential_1_batch_normalization_8_cast_2_readvariableop_resource:	?P
Asequential_1_batch_normalization_8_cast_3_readvariableop_resource:	?G
4sequential_1_dense_10_matmul_readvariableop_resource:	?	C
5sequential_1_dense_10_biasadd_readvariableop_resource:	
identity??7sequential_1/batch_normalization_10/Cast/ReadVariableOp?9sequential_1/batch_normalization_10/Cast_1/ReadVariableOp?9sequential_1/batch_normalization_10/Cast_2/ReadVariableOp?9sequential_1/batch_normalization_10/Cast_3/ReadVariableOp?7sequential_1/batch_normalization_11/Cast/ReadVariableOp?9sequential_1/batch_normalization_11/Cast_1/ReadVariableOp?9sequential_1/batch_normalization_11/Cast_2/ReadVariableOp?9sequential_1/batch_normalization_11/Cast_3/ReadVariableOp?6sequential_1/batch_normalization_8/Cast/ReadVariableOp?8sequential_1/batch_normalization_8/Cast_1/ReadVariableOp?8sequential_1/batch_normalization_8/Cast_2/ReadVariableOp?8sequential_1/batch_normalization_8/Cast_3/ReadVariableOp?6sequential_1/batch_normalization_9/Cast/ReadVariableOp?8sequential_1/batch_normalization_9/Cast_1/ReadVariableOp?8sequential_1/batch_normalization_9/Cast_2/ReadVariableOp?8sequential_1/batch_normalization_9/Cast_3/ReadVariableOp?,sequential_1/dense_10/BiasAdd/ReadVariableOp?+sequential_1/dense_10/MatMul/ReadVariableOp?,sequential_1/dense_11/BiasAdd/ReadVariableOp?+sequential_1/dense_11/MatMul/ReadVariableOp?,sequential_1/dense_12/BiasAdd/ReadVariableOp?+sequential_1/dense_12/MatMul/ReadVariableOp?,sequential_1/dense_13/BiasAdd/ReadVariableOp?+sequential_1/dense_13/MatMul/ReadVariableOp?,sequential_1/dense_14/BiasAdd/ReadVariableOp?+sequential_1/dense_14/MatMul/ReadVariableOp?
4sequential_1/dense_features_1/category_embed_1/ShapeShapecategory_embed_1*
T0*
_output_shapes
:26
4sequential_1/dense_features_1/category_embed_1/Shape?
Bsequential_1/dense_features_1/category_embed_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bsequential_1/dense_features_1/category_embed_1/strided_slice/stack?
Dsequential_1/dense_features_1/category_embed_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_1/dense_features_1/category_embed_1/strided_slice/stack_1?
Dsequential_1/dense_features_1/category_embed_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_1/dense_features_1/category_embed_1/strided_slice/stack_2?
<sequential_1/dense_features_1/category_embed_1/strided_sliceStridedSlice=sequential_1/dense_features_1/category_embed_1/Shape:output:0Ksequential_1/dense_features_1/category_embed_1/strided_slice/stack:output:0Msequential_1/dense_features_1/category_embed_1/strided_slice/stack_1:output:0Msequential_1/dense_features_1/category_embed_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<sequential_1/dense_features_1/category_embed_1/strided_slice?
>sequential_1/dense_features_1/category_embed_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2@
>sequential_1/dense_features_1/category_embed_1/Reshape/shape/1?
<sequential_1/dense_features_1/category_embed_1/Reshape/shapePackEsequential_1/dense_features_1/category_embed_1/strided_slice:output:0Gsequential_1/dense_features_1/category_embed_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2>
<sequential_1/dense_features_1/category_embed_1/Reshape/shape?
6sequential_1/dense_features_1/category_embed_1/ReshapeReshapecategory_embed_1Esequential_1/dense_features_1/category_embed_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????28
6sequential_1/dense_features_1/category_embed_1/Reshape?
4sequential_1/dense_features_1/category_embed_2/ShapeShapecategory_embed_2*
T0*
_output_shapes
:26
4sequential_1/dense_features_1/category_embed_2/Shape?
Bsequential_1/dense_features_1/category_embed_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bsequential_1/dense_features_1/category_embed_2/strided_slice/stack?
Dsequential_1/dense_features_1/category_embed_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_1/dense_features_1/category_embed_2/strided_slice/stack_1?
Dsequential_1/dense_features_1/category_embed_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_1/dense_features_1/category_embed_2/strided_slice/stack_2?
<sequential_1/dense_features_1/category_embed_2/strided_sliceStridedSlice=sequential_1/dense_features_1/category_embed_2/Shape:output:0Ksequential_1/dense_features_1/category_embed_2/strided_slice/stack:output:0Msequential_1/dense_features_1/category_embed_2/strided_slice/stack_1:output:0Msequential_1/dense_features_1/category_embed_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<sequential_1/dense_features_1/category_embed_2/strided_slice?
>sequential_1/dense_features_1/category_embed_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2@
>sequential_1/dense_features_1/category_embed_2/Reshape/shape/1?
<sequential_1/dense_features_1/category_embed_2/Reshape/shapePackEsequential_1/dense_features_1/category_embed_2/strided_slice:output:0Gsequential_1/dense_features_1/category_embed_2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2>
<sequential_1/dense_features_1/category_embed_2/Reshape/shape?
6sequential_1/dense_features_1/category_embed_2/ReshapeReshapecategory_embed_2Esequential_1/dense_features_1/category_embed_2/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????28
6sequential_1/dense_features_1/category_embed_2/Reshape?
4sequential_1/dense_features_1/category_embed_3/ShapeShapecategory_embed_3*
T0*
_output_shapes
:26
4sequential_1/dense_features_1/category_embed_3/Shape?
Bsequential_1/dense_features_1/category_embed_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bsequential_1/dense_features_1/category_embed_3/strided_slice/stack?
Dsequential_1/dense_features_1/category_embed_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_1/dense_features_1/category_embed_3/strided_slice/stack_1?
Dsequential_1/dense_features_1/category_embed_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_1/dense_features_1/category_embed_3/strided_slice/stack_2?
<sequential_1/dense_features_1/category_embed_3/strided_sliceStridedSlice=sequential_1/dense_features_1/category_embed_3/Shape:output:0Ksequential_1/dense_features_1/category_embed_3/strided_slice/stack:output:0Msequential_1/dense_features_1/category_embed_3/strided_slice/stack_1:output:0Msequential_1/dense_features_1/category_embed_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<sequential_1/dense_features_1/category_embed_3/strided_slice?
>sequential_1/dense_features_1/category_embed_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2@
>sequential_1/dense_features_1/category_embed_3/Reshape/shape/1?
<sequential_1/dense_features_1/category_embed_3/Reshape/shapePackEsequential_1/dense_features_1/category_embed_3/strided_slice:output:0Gsequential_1/dense_features_1/category_embed_3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2>
<sequential_1/dense_features_1/category_embed_3/Reshape/shape?
6sequential_1/dense_features_1/category_embed_3/ReshapeReshapecategory_embed_3Esequential_1/dense_features_1/category_embed_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????28
6sequential_1/dense_features_1/category_embed_3/Reshape?
4sequential_1/dense_features_1/category_embed_4/ShapeShapecategory_embed_4*
T0*
_output_shapes
:26
4sequential_1/dense_features_1/category_embed_4/Shape?
Bsequential_1/dense_features_1/category_embed_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bsequential_1/dense_features_1/category_embed_4/strided_slice/stack?
Dsequential_1/dense_features_1/category_embed_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_1/dense_features_1/category_embed_4/strided_slice/stack_1?
Dsequential_1/dense_features_1/category_embed_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_1/dense_features_1/category_embed_4/strided_slice/stack_2?
<sequential_1/dense_features_1/category_embed_4/strided_sliceStridedSlice=sequential_1/dense_features_1/category_embed_4/Shape:output:0Ksequential_1/dense_features_1/category_embed_4/strided_slice/stack:output:0Msequential_1/dense_features_1/category_embed_4/strided_slice/stack_1:output:0Msequential_1/dense_features_1/category_embed_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<sequential_1/dense_features_1/category_embed_4/strided_slice?
>sequential_1/dense_features_1/category_embed_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2@
>sequential_1/dense_features_1/category_embed_4/Reshape/shape/1?
<sequential_1/dense_features_1/category_embed_4/Reshape/shapePackEsequential_1/dense_features_1/category_embed_4/strided_slice:output:0Gsequential_1/dense_features_1/category_embed_4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2>
<sequential_1/dense_features_1/category_embed_4/Reshape/shape?
6sequential_1/dense_features_1/category_embed_4/ReshapeReshapecategory_embed_4Esequential_1/dense_features_1/category_embed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????28
6sequential_1/dense_features_1/category_embed_4/Reshape?
4sequential_1/dense_features_1/category_embed_5/ShapeShapecategory_embed_5*
T0*
_output_shapes
:26
4sequential_1/dense_features_1/category_embed_5/Shape?
Bsequential_1/dense_features_1/category_embed_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bsequential_1/dense_features_1/category_embed_5/strided_slice/stack?
Dsequential_1/dense_features_1/category_embed_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_1/dense_features_1/category_embed_5/strided_slice/stack_1?
Dsequential_1/dense_features_1/category_embed_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_1/dense_features_1/category_embed_5/strided_slice/stack_2?
<sequential_1/dense_features_1/category_embed_5/strided_sliceStridedSlice=sequential_1/dense_features_1/category_embed_5/Shape:output:0Ksequential_1/dense_features_1/category_embed_5/strided_slice/stack:output:0Msequential_1/dense_features_1/category_embed_5/strided_slice/stack_1:output:0Msequential_1/dense_features_1/category_embed_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<sequential_1/dense_features_1/category_embed_5/strided_slice?
>sequential_1/dense_features_1/category_embed_5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2@
>sequential_1/dense_features_1/category_embed_5/Reshape/shape/1?
<sequential_1/dense_features_1/category_embed_5/Reshape/shapePackEsequential_1/dense_features_1/category_embed_5/strided_slice:output:0Gsequential_1/dense_features_1/category_embed_5/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2>
<sequential_1/dense_features_1/category_embed_5/Reshape/shape?
6sequential_1/dense_features_1/category_embed_5/ReshapeReshapecategory_embed_5Esequential_1/dense_features_1/category_embed_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????28
6sequential_1/dense_features_1/category_embed_5/Reshape?
0sequential_1/dense_features_1/city_embed_1/ShapeShapecity_embed_1*
T0*
_output_shapes
:22
0sequential_1/dense_features_1/city_embed_1/Shape?
>sequential_1/dense_features_1/city_embed_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>sequential_1/dense_features_1/city_embed_1/strided_slice/stack?
@sequential_1/dense_features_1/city_embed_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_1/dense_features_1/city_embed_1/strided_slice/stack_1?
@sequential_1/dense_features_1/city_embed_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_1/dense_features_1/city_embed_1/strided_slice/stack_2?
8sequential_1/dense_features_1/city_embed_1/strided_sliceStridedSlice9sequential_1/dense_features_1/city_embed_1/Shape:output:0Gsequential_1/dense_features_1/city_embed_1/strided_slice/stack:output:0Isequential_1/dense_features_1/city_embed_1/strided_slice/stack_1:output:0Isequential_1/dense_features_1/city_embed_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2:
8sequential_1/dense_features_1/city_embed_1/strided_slice?
:sequential_1/dense_features_1/city_embed_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2<
:sequential_1/dense_features_1/city_embed_1/Reshape/shape/1?
8sequential_1/dense_features_1/city_embed_1/Reshape/shapePackAsequential_1/dense_features_1/city_embed_1/strided_slice:output:0Csequential_1/dense_features_1/city_embed_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2:
8sequential_1/dense_features_1/city_embed_1/Reshape/shape?
2sequential_1/dense_features_1/city_embed_1/ReshapeReshapecity_embed_1Asequential_1/dense_features_1/city_embed_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????24
2sequential_1/dense_features_1/city_embed_1/Reshape?
0sequential_1/dense_features_1/city_embed_2/ShapeShapecity_embed_2*
T0*
_output_shapes
:22
0sequential_1/dense_features_1/city_embed_2/Shape?
>sequential_1/dense_features_1/city_embed_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>sequential_1/dense_features_1/city_embed_2/strided_slice/stack?
@sequential_1/dense_features_1/city_embed_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_1/dense_features_1/city_embed_2/strided_slice/stack_1?
@sequential_1/dense_features_1/city_embed_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_1/dense_features_1/city_embed_2/strided_slice/stack_2?
8sequential_1/dense_features_1/city_embed_2/strided_sliceStridedSlice9sequential_1/dense_features_1/city_embed_2/Shape:output:0Gsequential_1/dense_features_1/city_embed_2/strided_slice/stack:output:0Isequential_1/dense_features_1/city_embed_2/strided_slice/stack_1:output:0Isequential_1/dense_features_1/city_embed_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2:
8sequential_1/dense_features_1/city_embed_2/strided_slice?
:sequential_1/dense_features_1/city_embed_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2<
:sequential_1/dense_features_1/city_embed_2/Reshape/shape/1?
8sequential_1/dense_features_1/city_embed_2/Reshape/shapePackAsequential_1/dense_features_1/city_embed_2/strided_slice:output:0Csequential_1/dense_features_1/city_embed_2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2:
8sequential_1/dense_features_1/city_embed_2/Reshape/shape?
2sequential_1/dense_features_1/city_embed_2/ReshapeReshapecity_embed_2Asequential_1/dense_features_1/city_embed_2/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????24
2sequential_1/dense_features_1/city_embed_2/Reshape?
0sequential_1/dense_features_1/city_embed_3/ShapeShapecity_embed_3*
T0*
_output_shapes
:22
0sequential_1/dense_features_1/city_embed_3/Shape?
>sequential_1/dense_features_1/city_embed_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>sequential_1/dense_features_1/city_embed_3/strided_slice/stack?
@sequential_1/dense_features_1/city_embed_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_1/dense_features_1/city_embed_3/strided_slice/stack_1?
@sequential_1/dense_features_1/city_embed_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_1/dense_features_1/city_embed_3/strided_slice/stack_2?
8sequential_1/dense_features_1/city_embed_3/strided_sliceStridedSlice9sequential_1/dense_features_1/city_embed_3/Shape:output:0Gsequential_1/dense_features_1/city_embed_3/strided_slice/stack:output:0Isequential_1/dense_features_1/city_embed_3/strided_slice/stack_1:output:0Isequential_1/dense_features_1/city_embed_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2:
8sequential_1/dense_features_1/city_embed_3/strided_slice?
:sequential_1/dense_features_1/city_embed_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2<
:sequential_1/dense_features_1/city_embed_3/Reshape/shape/1?
8sequential_1/dense_features_1/city_embed_3/Reshape/shapePackAsequential_1/dense_features_1/city_embed_3/strided_slice:output:0Csequential_1/dense_features_1/city_embed_3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2:
8sequential_1/dense_features_1/city_embed_3/Reshape/shape?
2sequential_1/dense_features_1/city_embed_3/ReshapeReshapecity_embed_3Asequential_1/dense_features_1/city_embed_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????24
2sequential_1/dense_features_1/city_embed_3/Reshape?
0sequential_1/dense_features_1/city_embed_4/ShapeShapecity_embed_4*
T0*
_output_shapes
:22
0sequential_1/dense_features_1/city_embed_4/Shape?
>sequential_1/dense_features_1/city_embed_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>sequential_1/dense_features_1/city_embed_4/strided_slice/stack?
@sequential_1/dense_features_1/city_embed_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_1/dense_features_1/city_embed_4/strided_slice/stack_1?
@sequential_1/dense_features_1/city_embed_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_1/dense_features_1/city_embed_4/strided_slice/stack_2?
8sequential_1/dense_features_1/city_embed_4/strided_sliceStridedSlice9sequential_1/dense_features_1/city_embed_4/Shape:output:0Gsequential_1/dense_features_1/city_embed_4/strided_slice/stack:output:0Isequential_1/dense_features_1/city_embed_4/strided_slice/stack_1:output:0Isequential_1/dense_features_1/city_embed_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2:
8sequential_1/dense_features_1/city_embed_4/strided_slice?
:sequential_1/dense_features_1/city_embed_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2<
:sequential_1/dense_features_1/city_embed_4/Reshape/shape/1?
8sequential_1/dense_features_1/city_embed_4/Reshape/shapePackAsequential_1/dense_features_1/city_embed_4/strided_slice:output:0Csequential_1/dense_features_1/city_embed_4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2:
8sequential_1/dense_features_1/city_embed_4/Reshape/shape?
2sequential_1/dense_features_1/city_embed_4/ReshapeReshapecity_embed_4Asequential_1/dense_features_1/city_embed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????24
2sequential_1/dense_features_1/city_embed_4/Reshape?
0sequential_1/dense_features_1/city_embed_5/ShapeShapecity_embed_5*
T0*
_output_shapes
:22
0sequential_1/dense_features_1/city_embed_5/Shape?
>sequential_1/dense_features_1/city_embed_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>sequential_1/dense_features_1/city_embed_5/strided_slice/stack?
@sequential_1/dense_features_1/city_embed_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_1/dense_features_1/city_embed_5/strided_slice/stack_1?
@sequential_1/dense_features_1/city_embed_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_1/dense_features_1/city_embed_5/strided_slice/stack_2?
8sequential_1/dense_features_1/city_embed_5/strided_sliceStridedSlice9sequential_1/dense_features_1/city_embed_5/Shape:output:0Gsequential_1/dense_features_1/city_embed_5/strided_slice/stack:output:0Isequential_1/dense_features_1/city_embed_5/strided_slice/stack_1:output:0Isequential_1/dense_features_1/city_embed_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2:
8sequential_1/dense_features_1/city_embed_5/strided_slice?
:sequential_1/dense_features_1/city_embed_5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2<
:sequential_1/dense_features_1/city_embed_5/Reshape/shape/1?
8sequential_1/dense_features_1/city_embed_5/Reshape/shapePackAsequential_1/dense_features_1/city_embed_5/strided_slice:output:0Csequential_1/dense_features_1/city_embed_5/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2:
8sequential_1/dense_features_1/city_embed_5/Reshape/shape?
2sequential_1/dense_features_1/city_embed_5/ReshapeReshapecity_embed_5Asequential_1/dense_features_1/city_embed_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????24
2sequential_1/dense_features_1/city_embed_5/Reshape?
)sequential_1/dense_features_1/colon/ShapeShapecolon*
T0*
_output_shapes
:2+
)sequential_1/dense_features_1/colon/Shape?
7sequential_1/dense_features_1/colon/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7sequential_1/dense_features_1/colon/strided_slice/stack?
9sequential_1/dense_features_1/colon/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_1/dense_features_1/colon/strided_slice/stack_1?
9sequential_1/dense_features_1/colon/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_1/dense_features_1/colon/strided_slice/stack_2?
1sequential_1/dense_features_1/colon/strided_sliceStridedSlice2sequential_1/dense_features_1/colon/Shape:output:0@sequential_1/dense_features_1/colon/strided_slice/stack:output:0Bsequential_1/dense_features_1/colon/strided_slice/stack_1:output:0Bsequential_1/dense_features_1/colon/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1sequential_1/dense_features_1/colon/strided_slice?
3sequential_1/dense_features_1/colon/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :25
3sequential_1/dense_features_1/colon/Reshape/shape/1?
1sequential_1/dense_features_1/colon/Reshape/shapePack:sequential_1/dense_features_1/colon/strided_slice:output:0<sequential_1/dense_features_1/colon/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:23
1sequential_1/dense_features_1/colon/Reshape/shape?
+sequential_1/dense_features_1/colon/ReshapeReshapecolon:sequential_1/dense_features_1/colon/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2-
+sequential_1/dense_features_1/colon/Reshape?
*sequential_1/dense_features_1/commas/ShapeShapecommas*
T0*
_output_shapes
:2,
*sequential_1/dense_features_1/commas/Shape?
8sequential_1/dense_features_1/commas/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8sequential_1/dense_features_1/commas/strided_slice/stack?
:sequential_1/dense_features_1/commas/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:sequential_1/dense_features_1/commas/strided_slice/stack_1?
:sequential_1/dense_features_1/commas/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:sequential_1/dense_features_1/commas/strided_slice/stack_2?
2sequential_1/dense_features_1/commas/strided_sliceStridedSlice3sequential_1/dense_features_1/commas/Shape:output:0Asequential_1/dense_features_1/commas/strided_slice/stack:output:0Csequential_1/dense_features_1/commas/strided_slice/stack_1:output:0Csequential_1/dense_features_1/commas/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2sequential_1/dense_features_1/commas/strided_slice?
4sequential_1/dense_features_1/commas/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :26
4sequential_1/dense_features_1/commas/Reshape/shape/1?
2sequential_1/dense_features_1/commas/Reshape/shapePack;sequential_1/dense_features_1/commas/strided_slice:output:0=sequential_1/dense_features_1/commas/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:24
2sequential_1/dense_features_1/commas/Reshape/shape?
,sequential_1/dense_features_1/commas/ReshapeReshapecommas;sequential_1/dense_features_1/commas/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2.
,sequential_1/dense_features_1/commas/Reshape?
(sequential_1/dense_features_1/dash/ShapeShapedash*
T0*
_output_shapes
:2*
(sequential_1/dense_features_1/dash/Shape?
6sequential_1/dense_features_1/dash/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6sequential_1/dense_features_1/dash/strided_slice/stack?
8sequential_1/dense_features_1/dash/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_1/dense_features_1/dash/strided_slice/stack_1?
8sequential_1/dense_features_1/dash/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_1/dense_features_1/dash/strided_slice/stack_2?
0sequential_1/dense_features_1/dash/strided_sliceStridedSlice1sequential_1/dense_features_1/dash/Shape:output:0?sequential_1/dense_features_1/dash/strided_slice/stack:output:0Asequential_1/dense_features_1/dash/strided_slice/stack_1:output:0Asequential_1/dense_features_1/dash/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential_1/dense_features_1/dash/strided_slice?
2sequential_1/dense_features_1/dash/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :24
2sequential_1/dense_features_1/dash/Reshape/shape/1?
0sequential_1/dense_features_1/dash/Reshape/shapePack9sequential_1/dense_features_1/dash/strided_slice:output:0;sequential_1/dense_features_1/dash/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:22
0sequential_1/dense_features_1/dash/Reshape/shape?
*sequential_1/dense_features_1/dash/ReshapeReshapedash9sequential_1/dense_features_1/dash/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2,
*sequential_1/dense_features_1/dash/Reshape?
*sequential_1/dense_features_1/exclam/ShapeShapeexclam*
T0*
_output_shapes
:2,
*sequential_1/dense_features_1/exclam/Shape?
8sequential_1/dense_features_1/exclam/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8sequential_1/dense_features_1/exclam/strided_slice/stack?
:sequential_1/dense_features_1/exclam/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:sequential_1/dense_features_1/exclam/strided_slice/stack_1?
:sequential_1/dense_features_1/exclam/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:sequential_1/dense_features_1/exclam/strided_slice/stack_2?
2sequential_1/dense_features_1/exclam/strided_sliceStridedSlice3sequential_1/dense_features_1/exclam/Shape:output:0Asequential_1/dense_features_1/exclam/strided_slice/stack:output:0Csequential_1/dense_features_1/exclam/strided_slice/stack_1:output:0Csequential_1/dense_features_1/exclam/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2sequential_1/dense_features_1/exclam/strided_slice?
4sequential_1/dense_features_1/exclam/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :26
4sequential_1/dense_features_1/exclam/Reshape/shape/1?
2sequential_1/dense_features_1/exclam/Reshape/shapePack;sequential_1/dense_features_1/exclam/strided_slice:output:0=sequential_1/dense_features_1/exclam/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:24
2sequential_1/dense_features_1/exclam/Reshape/shape?
,sequential_1/dense_features_1/exclam/ReshapeReshapeexclam;sequential_1/dense_features_1/exclam/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2.
,sequential_1/dense_features_1/exclam/Reshape?
)sequential_1/dense_features_1/money/ShapeShapemoney*
T0*
_output_shapes
:2+
)sequential_1/dense_features_1/money/Shape?
7sequential_1/dense_features_1/money/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7sequential_1/dense_features_1/money/strided_slice/stack?
9sequential_1/dense_features_1/money/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_1/dense_features_1/money/strided_slice/stack_1?
9sequential_1/dense_features_1/money/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_1/dense_features_1/money/strided_slice/stack_2?
1sequential_1/dense_features_1/money/strided_sliceStridedSlice2sequential_1/dense_features_1/money/Shape:output:0@sequential_1/dense_features_1/money/strided_slice/stack:output:0Bsequential_1/dense_features_1/money/strided_slice/stack_1:output:0Bsequential_1/dense_features_1/money/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1sequential_1/dense_features_1/money/strided_slice?
3sequential_1/dense_features_1/money/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :25
3sequential_1/dense_features_1/money/Reshape/shape/1?
1sequential_1/dense_features_1/money/Reshape/shapePack:sequential_1/dense_features_1/money/strided_slice:output:0<sequential_1/dense_features_1/money/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:23
1sequential_1/dense_features_1/money/Reshape/shape?
+sequential_1/dense_features_1/money/ReshapeReshapemoney:sequential_1/dense_features_1/money/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2-
+sequential_1/dense_features_1/money/Reshape?
)sequential_1/dense_features_1/month/ShapeShapemonth*
T0*
_output_shapes
:2+
)sequential_1/dense_features_1/month/Shape?
7sequential_1/dense_features_1/month/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7sequential_1/dense_features_1/month/strided_slice/stack?
9sequential_1/dense_features_1/month/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_1/dense_features_1/month/strided_slice/stack_1?
9sequential_1/dense_features_1/month/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_1/dense_features_1/month/strided_slice/stack_2?
1sequential_1/dense_features_1/month/strided_sliceStridedSlice2sequential_1/dense_features_1/month/Shape:output:0@sequential_1/dense_features_1/month/strided_slice/stack:output:0Bsequential_1/dense_features_1/month/strided_slice/stack_1:output:0Bsequential_1/dense_features_1/month/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1sequential_1/dense_features_1/month/strided_slice?
3sequential_1/dense_features_1/month/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :25
3sequential_1/dense_features_1/month/Reshape/shape/1?
1sequential_1/dense_features_1/month/Reshape/shapePack:sequential_1/dense_features_1/month/strided_slice:output:0<sequential_1/dense_features_1/month/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:23
1sequential_1/dense_features_1/month/Reshape/shape?
+sequential_1/dense_features_1/month/ReshapeReshapemonth:sequential_1/dense_features_1/month/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2-
+sequential_1/dense_features_1/month/Reshape?
/sequential_1/dense_features_1/parenthesis/ShapeShapeparenthesis*
T0*
_output_shapes
:21
/sequential_1/dense_features_1/parenthesis/Shape?
=sequential_1/dense_features_1/parenthesis/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=sequential_1/dense_features_1/parenthesis/strided_slice/stack?
?sequential_1/dense_features_1/parenthesis/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?sequential_1/dense_features_1/parenthesis/strided_slice/stack_1?
?sequential_1/dense_features_1/parenthesis/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?sequential_1/dense_features_1/parenthesis/strided_slice/stack_2?
7sequential_1/dense_features_1/parenthesis/strided_sliceStridedSlice8sequential_1/dense_features_1/parenthesis/Shape:output:0Fsequential_1/dense_features_1/parenthesis/strided_slice/stack:output:0Hsequential_1/dense_features_1/parenthesis/strided_slice/stack_1:output:0Hsequential_1/dense_features_1/parenthesis/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask29
7sequential_1/dense_features_1/parenthesis/strided_slice?
9sequential_1/dense_features_1/parenthesis/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2;
9sequential_1/dense_features_1/parenthesis/Reshape/shape/1?
7sequential_1/dense_features_1/parenthesis/Reshape/shapePack@sequential_1/dense_features_1/parenthesis/strided_slice:output:0Bsequential_1/dense_features_1/parenthesis/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:29
7sequential_1/dense_features_1/parenthesis/Reshape/shape?
1sequential_1/dense_features_1/parenthesis/ReshapeReshapeparenthesis@sequential_1/dense_features_1/parenthesis/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????23
1sequential_1/dense_features_1/parenthesis/Reshape?
1sequential_1/dense_features_1/state_embed_1/ShapeShapestate_embed_1*
T0*
_output_shapes
:23
1sequential_1/dense_features_1/state_embed_1/Shape?
?sequential_1/dense_features_1/state_embed_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?sequential_1/dense_features_1/state_embed_1/strided_slice/stack?
Asequential_1/dense_features_1/state_embed_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
Asequential_1/dense_features_1/state_embed_1/strided_slice/stack_1?
Asequential_1/dense_features_1/state_embed_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
Asequential_1/dense_features_1/state_embed_1/strided_slice/stack_2?
9sequential_1/dense_features_1/state_embed_1/strided_sliceStridedSlice:sequential_1/dense_features_1/state_embed_1/Shape:output:0Hsequential_1/dense_features_1/state_embed_1/strided_slice/stack:output:0Jsequential_1/dense_features_1/state_embed_1/strided_slice/stack_1:output:0Jsequential_1/dense_features_1/state_embed_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2;
9sequential_1/dense_features_1/state_embed_1/strided_slice?
;sequential_1/dense_features_1/state_embed_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2=
;sequential_1/dense_features_1/state_embed_1/Reshape/shape/1?
9sequential_1/dense_features_1/state_embed_1/Reshape/shapePackBsequential_1/dense_features_1/state_embed_1/strided_slice:output:0Dsequential_1/dense_features_1/state_embed_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2;
9sequential_1/dense_features_1/state_embed_1/Reshape/shape?
3sequential_1/dense_features_1/state_embed_1/ReshapeReshapestate_embed_1Bsequential_1/dense_features_1/state_embed_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????25
3sequential_1/dense_features_1/state_embed_1/Reshape?
1sequential_1/dense_features_1/state_embed_2/ShapeShapestate_embed_2*
T0*
_output_shapes
:23
1sequential_1/dense_features_1/state_embed_2/Shape?
?sequential_1/dense_features_1/state_embed_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?sequential_1/dense_features_1/state_embed_2/strided_slice/stack?
Asequential_1/dense_features_1/state_embed_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
Asequential_1/dense_features_1/state_embed_2/strided_slice/stack_1?
Asequential_1/dense_features_1/state_embed_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
Asequential_1/dense_features_1/state_embed_2/strided_slice/stack_2?
9sequential_1/dense_features_1/state_embed_2/strided_sliceStridedSlice:sequential_1/dense_features_1/state_embed_2/Shape:output:0Hsequential_1/dense_features_1/state_embed_2/strided_slice/stack:output:0Jsequential_1/dense_features_1/state_embed_2/strided_slice/stack_1:output:0Jsequential_1/dense_features_1/state_embed_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2;
9sequential_1/dense_features_1/state_embed_2/strided_slice?
;sequential_1/dense_features_1/state_embed_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2=
;sequential_1/dense_features_1/state_embed_2/Reshape/shape/1?
9sequential_1/dense_features_1/state_embed_2/Reshape/shapePackBsequential_1/dense_features_1/state_embed_2/strided_slice:output:0Dsequential_1/dense_features_1/state_embed_2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2;
9sequential_1/dense_features_1/state_embed_2/Reshape/shape?
3sequential_1/dense_features_1/state_embed_2/ReshapeReshapestate_embed_2Bsequential_1/dense_features_1/state_embed_2/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????25
3sequential_1/dense_features_1/state_embed_2/Reshape?
1sequential_1/dense_features_1/state_embed_3/ShapeShapestate_embed_3*
T0*
_output_shapes
:23
1sequential_1/dense_features_1/state_embed_3/Shape?
?sequential_1/dense_features_1/state_embed_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?sequential_1/dense_features_1/state_embed_3/strided_slice/stack?
Asequential_1/dense_features_1/state_embed_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
Asequential_1/dense_features_1/state_embed_3/strided_slice/stack_1?
Asequential_1/dense_features_1/state_embed_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
Asequential_1/dense_features_1/state_embed_3/strided_slice/stack_2?
9sequential_1/dense_features_1/state_embed_3/strided_sliceStridedSlice:sequential_1/dense_features_1/state_embed_3/Shape:output:0Hsequential_1/dense_features_1/state_embed_3/strided_slice/stack:output:0Jsequential_1/dense_features_1/state_embed_3/strided_slice/stack_1:output:0Jsequential_1/dense_features_1/state_embed_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2;
9sequential_1/dense_features_1/state_embed_3/strided_slice?
;sequential_1/dense_features_1/state_embed_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2=
;sequential_1/dense_features_1/state_embed_3/Reshape/shape/1?
9sequential_1/dense_features_1/state_embed_3/Reshape/shapePackBsequential_1/dense_features_1/state_embed_3/strided_slice:output:0Dsequential_1/dense_features_1/state_embed_3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2;
9sequential_1/dense_features_1/state_embed_3/Reshape/shape?
3sequential_1/dense_features_1/state_embed_3/ReshapeReshapestate_embed_3Bsequential_1/dense_features_1/state_embed_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????25
3sequential_1/dense_features_1/state_embed_3/Reshape?
1sequential_1/dense_features_1/state_embed_4/ShapeShapestate_embed_4*
T0*
_output_shapes
:23
1sequential_1/dense_features_1/state_embed_4/Shape?
?sequential_1/dense_features_1/state_embed_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?sequential_1/dense_features_1/state_embed_4/strided_slice/stack?
Asequential_1/dense_features_1/state_embed_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
Asequential_1/dense_features_1/state_embed_4/strided_slice/stack_1?
Asequential_1/dense_features_1/state_embed_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
Asequential_1/dense_features_1/state_embed_4/strided_slice/stack_2?
9sequential_1/dense_features_1/state_embed_4/strided_sliceStridedSlice:sequential_1/dense_features_1/state_embed_4/Shape:output:0Hsequential_1/dense_features_1/state_embed_4/strided_slice/stack:output:0Jsequential_1/dense_features_1/state_embed_4/strided_slice/stack_1:output:0Jsequential_1/dense_features_1/state_embed_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2;
9sequential_1/dense_features_1/state_embed_4/strided_slice?
;sequential_1/dense_features_1/state_embed_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2=
;sequential_1/dense_features_1/state_embed_4/Reshape/shape/1?
9sequential_1/dense_features_1/state_embed_4/Reshape/shapePackBsequential_1/dense_features_1/state_embed_4/strided_slice:output:0Dsequential_1/dense_features_1/state_embed_4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2;
9sequential_1/dense_features_1/state_embed_4/Reshape/shape?
3sequential_1/dense_features_1/state_embed_4/ReshapeReshapestate_embed_4Bsequential_1/dense_features_1/state_embed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????25
3sequential_1/dense_features_1/state_embed_4/Reshape?
1sequential_1/dense_features_1/state_embed_5/ShapeShapestate_embed_5*
T0*
_output_shapes
:23
1sequential_1/dense_features_1/state_embed_5/Shape?
?sequential_1/dense_features_1/state_embed_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?sequential_1/dense_features_1/state_embed_5/strided_slice/stack?
Asequential_1/dense_features_1/state_embed_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
Asequential_1/dense_features_1/state_embed_5/strided_slice/stack_1?
Asequential_1/dense_features_1/state_embed_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
Asequential_1/dense_features_1/state_embed_5/strided_slice/stack_2?
9sequential_1/dense_features_1/state_embed_5/strided_sliceStridedSlice:sequential_1/dense_features_1/state_embed_5/Shape:output:0Hsequential_1/dense_features_1/state_embed_5/strided_slice/stack:output:0Jsequential_1/dense_features_1/state_embed_5/strided_slice/stack_1:output:0Jsequential_1/dense_features_1/state_embed_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2;
9sequential_1/dense_features_1/state_embed_5/strided_slice?
;sequential_1/dense_features_1/state_embed_5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2=
;sequential_1/dense_features_1/state_embed_5/Reshape/shape/1?
9sequential_1/dense_features_1/state_embed_5/Reshape/shapePackBsequential_1/dense_features_1/state_embed_5/strided_slice:output:0Dsequential_1/dense_features_1/state_embed_5/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2;
9sequential_1/dense_features_1/state_embed_5/Reshape/shape?
3sequential_1/dense_features_1/state_embed_5/ReshapeReshapestate_embed_5Bsequential_1/dense_features_1/state_embed_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????25
3sequential_1/dense_features_1/state_embed_5/Reshape?
+sequential_1/dense_features_1/weekday/ShapeShapeweekday*
T0*
_output_shapes
:2-
+sequential_1/dense_features_1/weekday/Shape?
9sequential_1/dense_features_1/weekday/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9sequential_1/dense_features_1/weekday/strided_slice/stack?
;sequential_1/dense_features_1/weekday/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;sequential_1/dense_features_1/weekday/strided_slice/stack_1?
;sequential_1/dense_features_1/weekday/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;sequential_1/dense_features_1/weekday/strided_slice/stack_2?
3sequential_1/dense_features_1/weekday/strided_sliceStridedSlice4sequential_1/dense_features_1/weekday/Shape:output:0Bsequential_1/dense_features_1/weekday/strided_slice/stack:output:0Dsequential_1/dense_features_1/weekday/strided_slice/stack_1:output:0Dsequential_1/dense_features_1/weekday/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask25
3sequential_1/dense_features_1/weekday/strided_slice?
5sequential_1/dense_features_1/weekday/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :27
5sequential_1/dense_features_1/weekday/Reshape/shape/1?
3sequential_1/dense_features_1/weekday/Reshape/shapePack<sequential_1/dense_features_1/weekday/strided_slice:output:0>sequential_1/dense_features_1/weekday/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:25
3sequential_1/dense_features_1/weekday/Reshape/shape?
-sequential_1/dense_features_1/weekday/ReshapeReshapeweekday<sequential_1/dense_features_1/weekday/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2/
-sequential_1/dense_features_1/weekday/Reshape?
)sequential_1/dense_features_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)sequential_1/dense_features_1/concat/axis?
$sequential_1/dense_features_1/concatConcatV2?sequential_1/dense_features_1/category_embed_1/Reshape:output:0?sequential_1/dense_features_1/category_embed_2/Reshape:output:0?sequential_1/dense_features_1/category_embed_3/Reshape:output:0?sequential_1/dense_features_1/category_embed_4/Reshape:output:0?sequential_1/dense_features_1/category_embed_5/Reshape:output:0;sequential_1/dense_features_1/city_embed_1/Reshape:output:0;sequential_1/dense_features_1/city_embed_2/Reshape:output:0;sequential_1/dense_features_1/city_embed_3/Reshape:output:0;sequential_1/dense_features_1/city_embed_4/Reshape:output:0;sequential_1/dense_features_1/city_embed_5/Reshape:output:04sequential_1/dense_features_1/colon/Reshape:output:05sequential_1/dense_features_1/commas/Reshape:output:03sequential_1/dense_features_1/dash/Reshape:output:05sequential_1/dense_features_1/exclam/Reshape:output:04sequential_1/dense_features_1/money/Reshape:output:04sequential_1/dense_features_1/month/Reshape:output:0:sequential_1/dense_features_1/parenthesis/Reshape:output:0<sequential_1/dense_features_1/state_embed_1/Reshape:output:0<sequential_1/dense_features_1/state_embed_2/Reshape:output:0<sequential_1/dense_features_1/state_embed_3/Reshape:output:0<sequential_1/dense_features_1/state_embed_4/Reshape:output:0<sequential_1/dense_features_1/state_embed_5/Reshape:output:06sequential_1/dense_features_1/weekday/Reshape:output:02sequential_1/dense_features_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2&
$sequential_1/dense_features_1/concat?
+sequential_1/dense_14/MatMul/ReadVariableOpReadVariableOp4sequential_1_dense_14_matmul_readvariableop_resource*
_output_shapes

: *
dtype02-
+sequential_1/dense_14/MatMul/ReadVariableOp?
sequential_1/dense_14/MatMulMatMul-sequential_1/dense_features_1/concat:output:03sequential_1/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
sequential_1/dense_14/MatMul?
,sequential_1/dense_14/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_dense_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_1/dense_14/BiasAdd/ReadVariableOp?
sequential_1/dense_14/BiasAddBiasAdd&sequential_1/dense_14/MatMul:product:04sequential_1/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
sequential_1/dense_14/BiasAdd?
sequential_1/dense_14/ReluRelu&sequential_1/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
sequential_1/dense_14/Relu?
7sequential_1/batch_normalization_11/Cast/ReadVariableOpReadVariableOp@sequential_1_batch_normalization_11_cast_readvariableop_resource*
_output_shapes
: *
dtype029
7sequential_1/batch_normalization_11/Cast/ReadVariableOp?
9sequential_1/batch_normalization_11/Cast_1/ReadVariableOpReadVariableOpBsequential_1_batch_normalization_11_cast_1_readvariableop_resource*
_output_shapes
: *
dtype02;
9sequential_1/batch_normalization_11/Cast_1/ReadVariableOp?
9sequential_1/batch_normalization_11/Cast_2/ReadVariableOpReadVariableOpBsequential_1_batch_normalization_11_cast_2_readvariableop_resource*
_output_shapes
: *
dtype02;
9sequential_1/batch_normalization_11/Cast_2/ReadVariableOp?
9sequential_1/batch_normalization_11/Cast_3/ReadVariableOpReadVariableOpBsequential_1_batch_normalization_11_cast_3_readvariableop_resource*
_output_shapes
: *
dtype02;
9sequential_1/batch_normalization_11/Cast_3/ReadVariableOp?
3sequential_1/batch_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:25
3sequential_1/batch_normalization_11/batchnorm/add/y?
1sequential_1/batch_normalization_11/batchnorm/addAddV2Asequential_1/batch_normalization_11/Cast_1/ReadVariableOp:value:0<sequential_1/batch_normalization_11/batchnorm/add/y:output:0*
T0*
_output_shapes
: 23
1sequential_1/batch_normalization_11/batchnorm/add?
3sequential_1/batch_normalization_11/batchnorm/RsqrtRsqrt5sequential_1/batch_normalization_11/batchnorm/add:z:0*
T0*
_output_shapes
: 25
3sequential_1/batch_normalization_11/batchnorm/Rsqrt?
1sequential_1/batch_normalization_11/batchnorm/mulMul7sequential_1/batch_normalization_11/batchnorm/Rsqrt:y:0Asequential_1/batch_normalization_11/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
: 23
1sequential_1/batch_normalization_11/batchnorm/mul?
3sequential_1/batch_normalization_11/batchnorm/mul_1Mul(sequential_1/dense_14/Relu:activations:05sequential_1/batch_normalization_11/batchnorm/mul:z:0*
T0*'
_output_shapes
:????????? 25
3sequential_1/batch_normalization_11/batchnorm/mul_1?
3sequential_1/batch_normalization_11/batchnorm/mul_2Mul?sequential_1/batch_normalization_11/Cast/ReadVariableOp:value:05sequential_1/batch_normalization_11/batchnorm/mul:z:0*
T0*
_output_shapes
: 25
3sequential_1/batch_normalization_11/batchnorm/mul_2?
1sequential_1/batch_normalization_11/batchnorm/subSubAsequential_1/batch_normalization_11/Cast_2/ReadVariableOp:value:07sequential_1/batch_normalization_11/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 23
1sequential_1/batch_normalization_11/batchnorm/sub?
3sequential_1/batch_normalization_11/batchnorm/add_1AddV27sequential_1/batch_normalization_11/batchnorm/mul_1:z:05sequential_1/batch_normalization_11/batchnorm/sub:z:0*
T0*'
_output_shapes
:????????? 25
3sequential_1/batch_normalization_11/batchnorm/add_1?
+sequential_1/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_1_dense_13_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02-
+sequential_1/dense_13/MatMul/ReadVariableOp?
sequential_1/dense_13/MatMulMatMul7sequential_1/batch_normalization_11/batchnorm/add_1:z:03sequential_1/dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_1/dense_13/MatMul?
,sequential_1/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,sequential_1/dense_13/BiasAdd/ReadVariableOp?
sequential_1/dense_13/BiasAddBiasAdd&sequential_1/dense_13/MatMul:product:04sequential_1/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_1/dense_13/BiasAdd?
sequential_1/dense_13/ReluRelu&sequential_1/dense_13/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_1/dense_13/Relu?
7sequential_1/batch_normalization_10/Cast/ReadVariableOpReadVariableOp@sequential_1_batch_normalization_10_cast_readvariableop_resource*
_output_shapes	
:?*
dtype029
7sequential_1/batch_normalization_10/Cast/ReadVariableOp?
9sequential_1/batch_normalization_10/Cast_1/ReadVariableOpReadVariableOpBsequential_1_batch_normalization_10_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02;
9sequential_1/batch_normalization_10/Cast_1/ReadVariableOp?
9sequential_1/batch_normalization_10/Cast_2/ReadVariableOpReadVariableOpBsequential_1_batch_normalization_10_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype02;
9sequential_1/batch_normalization_10/Cast_2/ReadVariableOp?
9sequential_1/batch_normalization_10/Cast_3/ReadVariableOpReadVariableOpBsequential_1_batch_normalization_10_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype02;
9sequential_1/batch_normalization_10/Cast_3/ReadVariableOp?
3sequential_1/batch_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:25
3sequential_1/batch_normalization_10/batchnorm/add/y?
1sequential_1/batch_normalization_10/batchnorm/addAddV2Asequential_1/batch_normalization_10/Cast_1/ReadVariableOp:value:0<sequential_1/batch_normalization_10/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?23
1sequential_1/batch_normalization_10/batchnorm/add?
3sequential_1/batch_normalization_10/batchnorm/RsqrtRsqrt5sequential_1/batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes	
:?25
3sequential_1/batch_normalization_10/batchnorm/Rsqrt?
1sequential_1/batch_normalization_10/batchnorm/mulMul7sequential_1/batch_normalization_10/batchnorm/Rsqrt:y:0Asequential_1/batch_normalization_10/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?23
1sequential_1/batch_normalization_10/batchnorm/mul?
3sequential_1/batch_normalization_10/batchnorm/mul_1Mul(sequential_1/dense_13/Relu:activations:05sequential_1/batch_normalization_10/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????25
3sequential_1/batch_normalization_10/batchnorm/mul_1?
3sequential_1/batch_normalization_10/batchnorm/mul_2Mul?sequential_1/batch_normalization_10/Cast/ReadVariableOp:value:05sequential_1/batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes	
:?25
3sequential_1/batch_normalization_10/batchnorm/mul_2?
1sequential_1/batch_normalization_10/batchnorm/subSubAsequential_1/batch_normalization_10/Cast_2/ReadVariableOp:value:07sequential_1/batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?23
1sequential_1/batch_normalization_10/batchnorm/sub?
3sequential_1/batch_normalization_10/batchnorm/add_1AddV27sequential_1/batch_normalization_10/batchnorm/mul_1:z:05sequential_1/batch_normalization_10/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????25
3sequential_1/batch_normalization_10/batchnorm/add_1?
+sequential_1/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_1_dense_12_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02-
+sequential_1/dense_12/MatMul/ReadVariableOp?
sequential_1/dense_12/MatMulMatMul7sequential_1/batch_normalization_10/batchnorm/add_1:z:03sequential_1/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential_1/dense_12/MatMul?
,sequential_1/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_1/dense_12/BiasAdd/ReadVariableOp?
sequential_1/dense_12/BiasAddBiasAdd&sequential_1/dense_12/MatMul:product:04sequential_1/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential_1/dense_12/BiasAdd?
sequential_1/dense_12/ReluRelu&sequential_1/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
sequential_1/dense_12/Relu?
6sequential_1/batch_normalization_9/Cast/ReadVariableOpReadVariableOp?sequential_1_batch_normalization_9_cast_readvariableop_resource*
_output_shapes
:@*
dtype028
6sequential_1/batch_normalization_9/Cast/ReadVariableOp?
8sequential_1/batch_normalization_9/Cast_1/ReadVariableOpReadVariableOpAsequential_1_batch_normalization_9_cast_1_readvariableop_resource*
_output_shapes
:@*
dtype02:
8sequential_1/batch_normalization_9/Cast_1/ReadVariableOp?
8sequential_1/batch_normalization_9/Cast_2/ReadVariableOpReadVariableOpAsequential_1_batch_normalization_9_cast_2_readvariableop_resource*
_output_shapes
:@*
dtype02:
8sequential_1/batch_normalization_9/Cast_2/ReadVariableOp?
8sequential_1/batch_normalization_9/Cast_3/ReadVariableOpReadVariableOpAsequential_1_batch_normalization_9_cast_3_readvariableop_resource*
_output_shapes
:@*
dtype02:
8sequential_1/batch_normalization_9/Cast_3/ReadVariableOp?
2sequential_1/batch_normalization_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:24
2sequential_1/batch_normalization_9/batchnorm/add/y?
0sequential_1/batch_normalization_9/batchnorm/addAddV2@sequential_1/batch_normalization_9/Cast_1/ReadVariableOp:value:0;sequential_1/batch_normalization_9/batchnorm/add/y:output:0*
T0*
_output_shapes
:@22
0sequential_1/batch_normalization_9/batchnorm/add?
2sequential_1/batch_normalization_9/batchnorm/RsqrtRsqrt4sequential_1/batch_normalization_9/batchnorm/add:z:0*
T0*
_output_shapes
:@24
2sequential_1/batch_normalization_9/batchnorm/Rsqrt?
0sequential_1/batch_normalization_9/batchnorm/mulMul6sequential_1/batch_normalization_9/batchnorm/Rsqrt:y:0@sequential_1/batch_normalization_9/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:@22
0sequential_1/batch_normalization_9/batchnorm/mul?
2sequential_1/batch_normalization_9/batchnorm/mul_1Mul(sequential_1/dense_12/Relu:activations:04sequential_1/batch_normalization_9/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????@24
2sequential_1/batch_normalization_9/batchnorm/mul_1?
2sequential_1/batch_normalization_9/batchnorm/mul_2Mul>sequential_1/batch_normalization_9/Cast/ReadVariableOp:value:04sequential_1/batch_normalization_9/batchnorm/mul:z:0*
T0*
_output_shapes
:@24
2sequential_1/batch_normalization_9/batchnorm/mul_2?
0sequential_1/batch_normalization_9/batchnorm/subSub@sequential_1/batch_normalization_9/Cast_2/ReadVariableOp:value:06sequential_1/batch_normalization_9/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@22
0sequential_1/batch_normalization_9/batchnorm/sub?
2sequential_1/batch_normalization_9/batchnorm/add_1AddV26sequential_1/batch_normalization_9/batchnorm/mul_1:z:04sequential_1/batch_normalization_9/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????@24
2sequential_1/batch_normalization_9/batchnorm/add_1?
+sequential_1/dense_11/MatMul/ReadVariableOpReadVariableOp4sequential_1_dense_11_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02-
+sequential_1/dense_11/MatMul/ReadVariableOp?
sequential_1/dense_11/MatMulMatMul6sequential_1/batch_normalization_9/batchnorm/add_1:z:03sequential_1/dense_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_1/dense_11/MatMul?
,sequential_1/dense_11/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_dense_11_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,sequential_1/dense_11/BiasAdd/ReadVariableOp?
sequential_1/dense_11/BiasAddBiasAdd&sequential_1/dense_11/MatMul:product:04sequential_1/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_1/dense_11/BiasAdd?
sequential_1/dense_11/ReluRelu&sequential_1/dense_11/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_1/dense_11/Relu?
6sequential_1/batch_normalization_8/Cast/ReadVariableOpReadVariableOp?sequential_1_batch_normalization_8_cast_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential_1/batch_normalization_8/Cast/ReadVariableOp?
8sequential_1/batch_normalization_8/Cast_1/ReadVariableOpReadVariableOpAsequential_1_batch_normalization_8_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8sequential_1/batch_normalization_8/Cast_1/ReadVariableOp?
8sequential_1/batch_normalization_8/Cast_2/ReadVariableOpReadVariableOpAsequential_1_batch_normalization_8_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8sequential_1/batch_normalization_8/Cast_2/ReadVariableOp?
8sequential_1/batch_normalization_8/Cast_3/ReadVariableOpReadVariableOpAsequential_1_batch_normalization_8_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8sequential_1/batch_normalization_8/Cast_3/ReadVariableOp?
2sequential_1/batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:24
2sequential_1/batch_normalization_8/batchnorm/add/y?
0sequential_1/batch_normalization_8/batchnorm/addAddV2@sequential_1/batch_normalization_8/Cast_1/ReadVariableOp:value:0;sequential_1/batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?22
0sequential_1/batch_normalization_8/batchnorm/add?
2sequential_1/batch_normalization_8/batchnorm/RsqrtRsqrt4sequential_1/batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes	
:?24
2sequential_1/batch_normalization_8/batchnorm/Rsqrt?
0sequential_1/batch_normalization_8/batchnorm/mulMul6sequential_1/batch_normalization_8/batchnorm/Rsqrt:y:0@sequential_1/batch_normalization_8/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?22
0sequential_1/batch_normalization_8/batchnorm/mul?
2sequential_1/batch_normalization_8/batchnorm/mul_1Mul(sequential_1/dense_11/Relu:activations:04sequential_1/batch_normalization_8/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????24
2sequential_1/batch_normalization_8/batchnorm/mul_1?
2sequential_1/batch_normalization_8/batchnorm/mul_2Mul>sequential_1/batch_normalization_8/Cast/ReadVariableOp:value:04sequential_1/batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes	
:?24
2sequential_1/batch_normalization_8/batchnorm/mul_2?
0sequential_1/batch_normalization_8/batchnorm/subSub@sequential_1/batch_normalization_8/Cast_2/ReadVariableOp:value:06sequential_1/batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?22
0sequential_1/batch_normalization_8/batchnorm/sub?
2sequential_1/batch_normalization_8/batchnorm/add_1AddV26sequential_1/batch_normalization_8/batchnorm/mul_1:z:04sequential_1/batch_normalization_8/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????24
2sequential_1/batch_normalization_8/batchnorm/add_1?
+sequential_1/dense_10/MatMul/ReadVariableOpReadVariableOp4sequential_1_dense_10_matmul_readvariableop_resource*
_output_shapes
:	?	*
dtype02-
+sequential_1/dense_10/MatMul/ReadVariableOp?
sequential_1/dense_10/MatMulMatMul6sequential_1/batch_normalization_8/batchnorm/add_1:z:03sequential_1/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
sequential_1/dense_10/MatMul?
,sequential_1/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_dense_10_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02.
,sequential_1/dense_10/BiasAdd/ReadVariableOp?
sequential_1/dense_10/BiasAddBiasAdd&sequential_1/dense_10/MatMul:product:04sequential_1/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
sequential_1/dense_10/BiasAdd?
sequential_1/dense_10/SoftmaxSoftmax&sequential_1/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:?????????	2
sequential_1/dense_10/Softmax?
IdentityIdentity'sequential_1/dense_10/Softmax:softmax:08^sequential_1/batch_normalization_10/Cast/ReadVariableOp:^sequential_1/batch_normalization_10/Cast_1/ReadVariableOp:^sequential_1/batch_normalization_10/Cast_2/ReadVariableOp:^sequential_1/batch_normalization_10/Cast_3/ReadVariableOp8^sequential_1/batch_normalization_11/Cast/ReadVariableOp:^sequential_1/batch_normalization_11/Cast_1/ReadVariableOp:^sequential_1/batch_normalization_11/Cast_2/ReadVariableOp:^sequential_1/batch_normalization_11/Cast_3/ReadVariableOp7^sequential_1/batch_normalization_8/Cast/ReadVariableOp9^sequential_1/batch_normalization_8/Cast_1/ReadVariableOp9^sequential_1/batch_normalization_8/Cast_2/ReadVariableOp9^sequential_1/batch_normalization_8/Cast_3/ReadVariableOp7^sequential_1/batch_normalization_9/Cast/ReadVariableOp9^sequential_1/batch_normalization_9/Cast_1/ReadVariableOp9^sequential_1/batch_normalization_9/Cast_2/ReadVariableOp9^sequential_1/batch_normalization_9/Cast_3/ReadVariableOp-^sequential_1/dense_10/BiasAdd/ReadVariableOp,^sequential_1/dense_10/MatMul/ReadVariableOp-^sequential_1/dense_11/BiasAdd/ReadVariableOp,^sequential_1/dense_11/MatMul/ReadVariableOp-^sequential_1/dense_12/BiasAdd/ReadVariableOp,^sequential_1/dense_12/MatMul/ReadVariableOp-^sequential_1/dense_13/BiasAdd/ReadVariableOp,^sequential_1/dense_13/MatMul/ReadVariableOp-^sequential_1/dense_14/BiasAdd/ReadVariableOp,^sequential_1/dense_14/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2r
7sequential_1/batch_normalization_10/Cast/ReadVariableOp7sequential_1/batch_normalization_10/Cast/ReadVariableOp2v
9sequential_1/batch_normalization_10/Cast_1/ReadVariableOp9sequential_1/batch_normalization_10/Cast_1/ReadVariableOp2v
9sequential_1/batch_normalization_10/Cast_2/ReadVariableOp9sequential_1/batch_normalization_10/Cast_2/ReadVariableOp2v
9sequential_1/batch_normalization_10/Cast_3/ReadVariableOp9sequential_1/batch_normalization_10/Cast_3/ReadVariableOp2r
7sequential_1/batch_normalization_11/Cast/ReadVariableOp7sequential_1/batch_normalization_11/Cast/ReadVariableOp2v
9sequential_1/batch_normalization_11/Cast_1/ReadVariableOp9sequential_1/batch_normalization_11/Cast_1/ReadVariableOp2v
9sequential_1/batch_normalization_11/Cast_2/ReadVariableOp9sequential_1/batch_normalization_11/Cast_2/ReadVariableOp2v
9sequential_1/batch_normalization_11/Cast_3/ReadVariableOp9sequential_1/batch_normalization_11/Cast_3/ReadVariableOp2p
6sequential_1/batch_normalization_8/Cast/ReadVariableOp6sequential_1/batch_normalization_8/Cast/ReadVariableOp2t
8sequential_1/batch_normalization_8/Cast_1/ReadVariableOp8sequential_1/batch_normalization_8/Cast_1/ReadVariableOp2t
8sequential_1/batch_normalization_8/Cast_2/ReadVariableOp8sequential_1/batch_normalization_8/Cast_2/ReadVariableOp2t
8sequential_1/batch_normalization_8/Cast_3/ReadVariableOp8sequential_1/batch_normalization_8/Cast_3/ReadVariableOp2p
6sequential_1/batch_normalization_9/Cast/ReadVariableOp6sequential_1/batch_normalization_9/Cast/ReadVariableOp2t
8sequential_1/batch_normalization_9/Cast_1/ReadVariableOp8sequential_1/batch_normalization_9/Cast_1/ReadVariableOp2t
8sequential_1/batch_normalization_9/Cast_2/ReadVariableOp8sequential_1/batch_normalization_9/Cast_2/ReadVariableOp2t
8sequential_1/batch_normalization_9/Cast_3/ReadVariableOp8sequential_1/batch_normalization_9/Cast_3/ReadVariableOp2\
,sequential_1/dense_10/BiasAdd/ReadVariableOp,sequential_1/dense_10/BiasAdd/ReadVariableOp2Z
+sequential_1/dense_10/MatMul/ReadVariableOp+sequential_1/dense_10/MatMul/ReadVariableOp2\
,sequential_1/dense_11/BiasAdd/ReadVariableOp,sequential_1/dense_11/BiasAdd/ReadVariableOp2Z
+sequential_1/dense_11/MatMul/ReadVariableOp+sequential_1/dense_11/MatMul/ReadVariableOp2\
,sequential_1/dense_12/BiasAdd/ReadVariableOp,sequential_1/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_1/dense_12/MatMul/ReadVariableOp+sequential_1/dense_12/MatMul/ReadVariableOp2\
,sequential_1/dense_13/BiasAdd/ReadVariableOp,sequential_1/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_1/dense_13/MatMul/ReadVariableOp+sequential_1/dense_13/MatMul/ReadVariableOp2\
,sequential_1/dense_14/BiasAdd/ReadVariableOp,sequential_1/dense_14/BiasAdd/ReadVariableOp2Z
+sequential_1/dense_14/MatMul/ReadVariableOp+sequential_1/dense_14/MatMul/ReadVariableOp:Y U
'
_output_shapes
:?????????
*
_user_specified_namecategory_embed_1:YU
'
_output_shapes
:?????????
*
_user_specified_namecategory_embed_2:YU
'
_output_shapes
:?????????
*
_user_specified_namecategory_embed_3:YU
'
_output_shapes
:?????????
*
_user_specified_namecategory_embed_4:YU
'
_output_shapes
:?????????
*
_user_specified_namecategory_embed_5:UQ
'
_output_shapes
:?????????
&
_user_specified_namecity_embed_1:UQ
'
_output_shapes
:?????????
&
_user_specified_namecity_embed_2:UQ
'
_output_shapes
:?????????
&
_user_specified_namecity_embed_3:UQ
'
_output_shapes
:?????????
&
_user_specified_namecity_embed_4:U	Q
'
_output_shapes
:?????????
&
_user_specified_namecity_embed_5:N
J
'
_output_shapes
:?????????

_user_specified_namecolon:OK
'
_output_shapes
:?????????
 
_user_specified_namecommas:MI
'
_output_shapes
:?????????

_user_specified_namedash:OK
'
_output_shapes
:?????????
 
_user_specified_nameexclam:NJ
'
_output_shapes
:?????????

_user_specified_namemoney:NJ
'
_output_shapes
:?????????

_user_specified_namemonth:TP
'
_output_shapes
:?????????
%
_user_specified_nameparenthesis:VR
'
_output_shapes
:?????????
'
_user_specified_namestate_embed_1:VR
'
_output_shapes
:?????????
'
_user_specified_namestate_embed_2:VR
'
_output_shapes
:?????????
'
_user_specified_namestate_embed_3:VR
'
_output_shapes
:?????????
'
_user_specified_namestate_embed_4:VR
'
_output_shapes
:?????????
'
_user_specified_namestate_embed_5:PL
'
_output_shapes
:?????????
!
_user_specified_name	weekday
?
?
)__inference_dense_12_layer_call_fn_564091

inputs
unknown:	?@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_5614472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?)
?
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_564288

inputs6
'assignmovingavg_readvariableop_resource:	?8
)assignmovingavg_1_readvariableop_resource:	?+
cast_readvariableop_resource:	?-
cast_1_readvariableop_resource:	?
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?)
?
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_561029

inputs6
'assignmovingavg_readvariableop_resource:	?8
)assignmovingavg_1_readvariableop_resource:	?+
cast_readvariableop_resource:	?-
cast_1_readvariableop_resource:	?
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?m
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_562327
category_embed_1
category_embed_2
category_embed_3
category_embed_4
category_embed_5
city_embed_1
city_embed_2
city_embed_3
city_embed_4
city_embed_5	
colon

commas
dash

exclam	
money	
month
parenthesis
state_embed_1
state_embed_2
state_embed_3
state_embed_4
state_embed_5
weekday!
dense_14_562253: 
dense_14_562255: +
batch_normalization_11_562258: +
batch_normalization_11_562260: +
batch_normalization_11_562262: +
batch_normalization_11_562264: "
dense_13_562267:	 ?
dense_13_562269:	?,
batch_normalization_10_562272:	?,
batch_normalization_10_562274:	?,
batch_normalization_10_562276:	?,
batch_normalization_10_562278:	?"
dense_12_562281:	?@
dense_12_562283:@*
batch_normalization_9_562286:@*
batch_normalization_9_562288:@*
batch_normalization_9_562290:@*
batch_normalization_9_562292:@"
dense_11_562295:	@?
dense_11_562297:	?+
batch_normalization_8_562300:	?+
batch_normalization_8_562302:	?+
batch_normalization_8_562304:	?+
batch_normalization_8_562306:	?"
dense_10_562309:	?	
dense_10_562311:	
identity??.batch_normalization_10/StatefulPartitionedCall?.batch_normalization_11/StatefulPartitionedCall?-batch_normalization_8/StatefulPartitionedCall?-batch_normalization_9/StatefulPartitionedCall? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall? dense_14/StatefulPartitionedCall?>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp?>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp?
 dense_features_1/PartitionedCallPartitionedCallcategory_embed_1category_embed_2category_embed_3category_embed_4category_embed_5city_embed_1city_embed_2city_embed_3city_embed_4city_embed_5coloncommasdashexclammoneymonthparenthesisstate_embed_1state_embed_2state_embed_3state_embed_4state_embed_5weekday*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_dense_features_1_layer_call_and_return_conditional_losses_5613702"
 dense_features_1/PartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_features_1/PartitionedCall:output:0dense_14_562253dense_14_562255*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_5613832"
 dense_14/StatefulPartitionedCall?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0batch_normalization_11_562258batch_normalization_11_562260batch_normalization_11_562262batch_normalization_11_562264*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_56048320
.batch_normalization_11/StatefulPartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0dense_13_562267dense_13_562269*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_5614152"
 dense_13/StatefulPartitionedCall?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0batch_normalization_10_562272batch_normalization_10_562274batch_normalization_10_562276batch_normalization_10_562278*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_56064520
.batch_normalization_10/StatefulPartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0dense_12_562281dense_12_562283*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_5614472"
 dense_12/StatefulPartitionedCall?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0batch_normalization_9_562286batch_normalization_9_562288batch_normalization_9_562290batch_normalization_9_562292*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_5608072/
-batch_normalization_9/StatefulPartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0dense_11_562295dense_11_562297*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_5614732"
 dense_11/StatefulPartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0batch_normalization_8_562300batch_normalization_8_562302batch_normalization_8_562304batch_normalization_8_562306*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_5609692/
-batch_normalization_8/StatefulPartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0dense_10_562309dense_10_562311*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_5614992"
 dense_10/StatefulPartitionedCall?
>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_13_562267*
_output_shapes
:	 ?*
dtype02@
>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp?
/sequential_1/dense_13/kernel/Regularizer/SquareSquareFsequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 ?21
/sequential_1/dense_13/kernel/Regularizer/Square?
.sequential_1/dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       20
.sequential_1/dense_13/kernel/Regularizer/Const?
,sequential_1/dense_13/kernel/Regularizer/SumSum3sequential_1/dense_13/kernel/Regularizer/Square:y:07sequential_1/dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2.
,sequential_1/dense_13/kernel/Regularizer/Sum?
.sequential_1/dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_1/dense_13/kernel/Regularizer/mul/x?
,sequential_1/dense_13/kernel/Regularizer/mulMul7sequential_1/dense_13/kernel/Regularizer/mul/x:output:05sequential_1/dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,sequential_1/dense_13/kernel/Regularizer/mul?
>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_12_562281*
_output_shapes
:	?@*
dtype02@
>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp?
/sequential_1/dense_12/kernel/Regularizer/SquareSquareFsequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?@21
/sequential_1/dense_12/kernel/Regularizer/Square?
.sequential_1/dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       20
.sequential_1/dense_12/kernel/Regularizer/Const?
,sequential_1/dense_12/kernel/Regularizer/SumSum3sequential_1/dense_12/kernel/Regularizer/Square:y:07sequential_1/dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2.
,sequential_1/dense_12/kernel/Regularizer/Sum?
.sequential_1/dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>20
.sequential_1/dense_12/kernel/Regularizer/mul/x?
,sequential_1/dense_12/kernel/Regularizer/mulMul7sequential_1/dense_12/kernel/Regularizer/mul/x:output:05sequential_1/dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,sequential_1/dense_12/kernel/Regularizer/mul?
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall?^sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp?^sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2?
>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp2?
>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp:Y U
'
_output_shapes
:?????????
*
_user_specified_namecategory_embed_1:YU
'
_output_shapes
:?????????
*
_user_specified_namecategory_embed_2:YU
'
_output_shapes
:?????????
*
_user_specified_namecategory_embed_3:YU
'
_output_shapes
:?????????
*
_user_specified_namecategory_embed_4:YU
'
_output_shapes
:?????????
*
_user_specified_namecategory_embed_5:UQ
'
_output_shapes
:?????????
&
_user_specified_namecity_embed_1:UQ
'
_output_shapes
:?????????
&
_user_specified_namecity_embed_2:UQ
'
_output_shapes
:?????????
&
_user_specified_namecity_embed_3:UQ
'
_output_shapes
:?????????
&
_user_specified_namecity_embed_4:U	Q
'
_output_shapes
:?????????
&
_user_specified_namecity_embed_5:N
J
'
_output_shapes
:?????????

_user_specified_namecolon:OK
'
_output_shapes
:?????????
 
_user_specified_namecommas:MI
'
_output_shapes
:?????????

_user_specified_namedash:OK
'
_output_shapes
:?????????
 
_user_specified_nameexclam:NJ
'
_output_shapes
:?????????

_user_specified_namemoney:NJ
'
_output_shapes
:?????????

_user_specified_namemonth:TP
'
_output_shapes
:?????????
%
_user_specified_nameparenthesis:VR
'
_output_shapes
:?????????
'
_user_specified_namestate_embed_1:VR
'
_output_shapes
:?????????
'
_user_specified_namestate_embed_2:VR
'
_output_shapes
:?????????
'
_user_specified_namestate_embed_3:VR
'
_output_shapes
:?????????
'
_user_specified_namestate_embed_4:VR
'
_output_shapes
:?????????
'
_user_specified_namestate_embed_5:PL
'
_output_shapes
:?????????
!
_user_specified_name	weekday
??
?&
__inference__traced_save_564588
file_prefix;
7savev2_sequential_1_dense_14_kernel_read_readvariableop9
5savev2_sequential_1_dense_14_bias_read_readvariableopH
Dsavev2_sequential_1_batch_normalization_11_gamma_read_readvariableopG
Csavev2_sequential_1_batch_normalization_11_beta_read_readvariableopN
Jsavev2_sequential_1_batch_normalization_11_moving_mean_read_readvariableopR
Nsavev2_sequential_1_batch_normalization_11_moving_variance_read_readvariableop;
7savev2_sequential_1_dense_13_kernel_read_readvariableop9
5savev2_sequential_1_dense_13_bias_read_readvariableopH
Dsavev2_sequential_1_batch_normalization_10_gamma_read_readvariableopG
Csavev2_sequential_1_batch_normalization_10_beta_read_readvariableopN
Jsavev2_sequential_1_batch_normalization_10_moving_mean_read_readvariableopR
Nsavev2_sequential_1_batch_normalization_10_moving_variance_read_readvariableop;
7savev2_sequential_1_dense_12_kernel_read_readvariableop9
5savev2_sequential_1_dense_12_bias_read_readvariableopG
Csavev2_sequential_1_batch_normalization_9_gamma_read_readvariableopF
Bsavev2_sequential_1_batch_normalization_9_beta_read_readvariableopM
Isavev2_sequential_1_batch_normalization_9_moving_mean_read_readvariableopQ
Msavev2_sequential_1_batch_normalization_9_moving_variance_read_readvariableop;
7savev2_sequential_1_dense_11_kernel_read_readvariableop9
5savev2_sequential_1_dense_11_bias_read_readvariableopG
Csavev2_sequential_1_batch_normalization_8_gamma_read_readvariableopF
Bsavev2_sequential_1_batch_normalization_8_beta_read_readvariableopM
Isavev2_sequential_1_batch_normalization_8_moving_mean_read_readvariableopQ
Msavev2_sequential_1_batch_normalization_8_moving_variance_read_readvariableop;
7savev2_sequential_1_dense_10_kernel_read_readvariableop9
5savev2_sequential_1_dense_10_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopB
>savev2_adam_sequential_1_dense_14_kernel_m_read_readvariableop@
<savev2_adam_sequential_1_dense_14_bias_m_read_readvariableopO
Ksavev2_adam_sequential_1_batch_normalization_11_gamma_m_read_readvariableopN
Jsavev2_adam_sequential_1_batch_normalization_11_beta_m_read_readvariableopB
>savev2_adam_sequential_1_dense_13_kernel_m_read_readvariableop@
<savev2_adam_sequential_1_dense_13_bias_m_read_readvariableopO
Ksavev2_adam_sequential_1_batch_normalization_10_gamma_m_read_readvariableopN
Jsavev2_adam_sequential_1_batch_normalization_10_beta_m_read_readvariableopB
>savev2_adam_sequential_1_dense_12_kernel_m_read_readvariableop@
<savev2_adam_sequential_1_dense_12_bias_m_read_readvariableopN
Jsavev2_adam_sequential_1_batch_normalization_9_gamma_m_read_readvariableopM
Isavev2_adam_sequential_1_batch_normalization_9_beta_m_read_readvariableopB
>savev2_adam_sequential_1_dense_11_kernel_m_read_readvariableop@
<savev2_adam_sequential_1_dense_11_bias_m_read_readvariableopN
Jsavev2_adam_sequential_1_batch_normalization_8_gamma_m_read_readvariableopM
Isavev2_adam_sequential_1_batch_normalization_8_beta_m_read_readvariableopB
>savev2_adam_sequential_1_dense_10_kernel_m_read_readvariableop@
<savev2_adam_sequential_1_dense_10_bias_m_read_readvariableopB
>savev2_adam_sequential_1_dense_14_kernel_v_read_readvariableop@
<savev2_adam_sequential_1_dense_14_bias_v_read_readvariableopO
Ksavev2_adam_sequential_1_batch_normalization_11_gamma_v_read_readvariableopN
Jsavev2_adam_sequential_1_batch_normalization_11_beta_v_read_readvariableopB
>savev2_adam_sequential_1_dense_13_kernel_v_read_readvariableop@
<savev2_adam_sequential_1_dense_13_bias_v_read_readvariableopO
Ksavev2_adam_sequential_1_batch_normalization_10_gamma_v_read_readvariableopN
Jsavev2_adam_sequential_1_batch_normalization_10_beta_v_read_readvariableopB
>savev2_adam_sequential_1_dense_12_kernel_v_read_readvariableop@
<savev2_adam_sequential_1_dense_12_bias_v_read_readvariableopN
Jsavev2_adam_sequential_1_batch_normalization_9_gamma_v_read_readvariableopM
Isavev2_adam_sequential_1_batch_normalization_9_beta_v_read_readvariableopB
>savev2_adam_sequential_1_dense_11_kernel_v_read_readvariableop@
<savev2_adam_sequential_1_dense_11_bias_v_read_readvariableopN
Jsavev2_adam_sequential_1_batch_normalization_8_gamma_v_read_readvariableopM
Isavev2_adam_sequential_1_batch_normalization_8_beta_v_read_readvariableopB
>savev2_adam_sequential_1_dense_10_kernel_v_read_readvariableop@
<savev2_adam_sequential_1_dense_10_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?'
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:H*
dtype0*?&
value?&B?&HB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:H*
dtype0*?
value?B?HB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?%
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:07savev2_sequential_1_dense_14_kernel_read_readvariableop5savev2_sequential_1_dense_14_bias_read_readvariableopDsavev2_sequential_1_batch_normalization_11_gamma_read_readvariableopCsavev2_sequential_1_batch_normalization_11_beta_read_readvariableopJsavev2_sequential_1_batch_normalization_11_moving_mean_read_readvariableopNsavev2_sequential_1_batch_normalization_11_moving_variance_read_readvariableop7savev2_sequential_1_dense_13_kernel_read_readvariableop5savev2_sequential_1_dense_13_bias_read_readvariableopDsavev2_sequential_1_batch_normalization_10_gamma_read_readvariableopCsavev2_sequential_1_batch_normalization_10_beta_read_readvariableopJsavev2_sequential_1_batch_normalization_10_moving_mean_read_readvariableopNsavev2_sequential_1_batch_normalization_10_moving_variance_read_readvariableop7savev2_sequential_1_dense_12_kernel_read_readvariableop5savev2_sequential_1_dense_12_bias_read_readvariableopCsavev2_sequential_1_batch_normalization_9_gamma_read_readvariableopBsavev2_sequential_1_batch_normalization_9_beta_read_readvariableopIsavev2_sequential_1_batch_normalization_9_moving_mean_read_readvariableopMsavev2_sequential_1_batch_normalization_9_moving_variance_read_readvariableop7savev2_sequential_1_dense_11_kernel_read_readvariableop5savev2_sequential_1_dense_11_bias_read_readvariableopCsavev2_sequential_1_batch_normalization_8_gamma_read_readvariableopBsavev2_sequential_1_batch_normalization_8_beta_read_readvariableopIsavev2_sequential_1_batch_normalization_8_moving_mean_read_readvariableopMsavev2_sequential_1_batch_normalization_8_moving_variance_read_readvariableop7savev2_sequential_1_dense_10_kernel_read_readvariableop5savev2_sequential_1_dense_10_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop>savev2_adam_sequential_1_dense_14_kernel_m_read_readvariableop<savev2_adam_sequential_1_dense_14_bias_m_read_readvariableopKsavev2_adam_sequential_1_batch_normalization_11_gamma_m_read_readvariableopJsavev2_adam_sequential_1_batch_normalization_11_beta_m_read_readvariableop>savev2_adam_sequential_1_dense_13_kernel_m_read_readvariableop<savev2_adam_sequential_1_dense_13_bias_m_read_readvariableopKsavev2_adam_sequential_1_batch_normalization_10_gamma_m_read_readvariableopJsavev2_adam_sequential_1_batch_normalization_10_beta_m_read_readvariableop>savev2_adam_sequential_1_dense_12_kernel_m_read_readvariableop<savev2_adam_sequential_1_dense_12_bias_m_read_readvariableopJsavev2_adam_sequential_1_batch_normalization_9_gamma_m_read_readvariableopIsavev2_adam_sequential_1_batch_normalization_9_beta_m_read_readvariableop>savev2_adam_sequential_1_dense_11_kernel_m_read_readvariableop<savev2_adam_sequential_1_dense_11_bias_m_read_readvariableopJsavev2_adam_sequential_1_batch_normalization_8_gamma_m_read_readvariableopIsavev2_adam_sequential_1_batch_normalization_8_beta_m_read_readvariableop>savev2_adam_sequential_1_dense_10_kernel_m_read_readvariableop<savev2_adam_sequential_1_dense_10_bias_m_read_readvariableop>savev2_adam_sequential_1_dense_14_kernel_v_read_readvariableop<savev2_adam_sequential_1_dense_14_bias_v_read_readvariableopKsavev2_adam_sequential_1_batch_normalization_11_gamma_v_read_readvariableopJsavev2_adam_sequential_1_batch_normalization_11_beta_v_read_readvariableop>savev2_adam_sequential_1_dense_13_kernel_v_read_readvariableop<savev2_adam_sequential_1_dense_13_bias_v_read_readvariableopKsavev2_adam_sequential_1_batch_normalization_10_gamma_v_read_readvariableopJsavev2_adam_sequential_1_batch_normalization_10_beta_v_read_readvariableop>savev2_adam_sequential_1_dense_12_kernel_v_read_readvariableop<savev2_adam_sequential_1_dense_12_bias_v_read_readvariableopJsavev2_adam_sequential_1_batch_normalization_9_gamma_v_read_readvariableopIsavev2_adam_sequential_1_batch_normalization_9_beta_v_read_readvariableop>savev2_adam_sequential_1_dense_11_kernel_v_read_readvariableop<savev2_adam_sequential_1_dense_11_bias_v_read_readvariableopJsavev2_adam_sequential_1_batch_normalization_8_gamma_v_read_readvariableopIsavev2_adam_sequential_1_batch_normalization_8_beta_v_read_readvariableop>savev2_adam_sequential_1_dense_10_kernel_v_read_readvariableop<savev2_adam_sequential_1_dense_10_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *V
dtypesL
J2H	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : : :	 ?:?:?:?:?:?:	?@:@:@:@:@:@:	@?:?:?:?:?:?:	?	:	: : : : : : : : : : : : : :	 ?:?:?:?:	?@:@:@:@:	@?:?:?:?:	?	:	: : : : :	 ?:?:?:?:	?@:@:@:@:	@?:?:?:?:	?	:	: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :%!

_output_shapes
:	 ?:!

_output_shapes	
:?:!	

_output_shapes	
:?:!


_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:%!

_output_shapes
:	?@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:%!

_output_shapes
:	@?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:%!

_output_shapes
:	?	: 

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$$ 

_output_shapes

: : %

_output_shapes
: : &

_output_shapes
: : '

_output_shapes
: :%(!

_output_shapes
:	 ?:!)

_output_shapes	
:?:!*

_output_shapes	
:?:!+

_output_shapes	
:?:%,!

_output_shapes
:	?@: -

_output_shapes
:@: .

_output_shapes
:@: /

_output_shapes
:@:%0!

_output_shapes
:	@?:!1

_output_shapes	
:?:!2

_output_shapes	
:?:!3

_output_shapes	
:?:%4!

_output_shapes
:	?	: 5

_output_shapes
:	:$6 

_output_shapes

: : 7

_output_shapes
: : 8

_output_shapes
: : 9

_output_shapes
: :%:!

_output_shapes
:	 ?:!;

_output_shapes	
:?:!<

_output_shapes	
:?:!=

_output_shapes	
:?:%>!

_output_shapes
:	?@: ?

_output_shapes
:@: @

_output_shapes
:@: A

_output_shapes
:@:%B!

_output_shapes
:	@?:!C

_output_shapes	
:?:!D

_output_shapes	
:?:!E

_output_shapes	
:?:%F!

_output_shapes
:	?	: G

_output_shapes
:	:H

_output_shapes
: 
?
?
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_560483

inputs*
cast_readvariableop_resource: ,
cast_1_readvariableop_resource: ,
cast_2_readvariableop_resource: ,
cast_3_readvariableop_resource: 
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
: *
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
: *
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
: *
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
: *
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:????????? 2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:????????? 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?)
?
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_560867

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@*
cast_readvariableop_resource:@,
cast_1_readvariableop_resource:@
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????@2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:@*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:@*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????@2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?4
"__inference__traced_restore_564811
file_prefix?
-assignvariableop_sequential_1_dense_14_kernel: ;
-assignvariableop_1_sequential_1_dense_14_bias: J
<assignvariableop_2_sequential_1_batch_normalization_11_gamma: I
;assignvariableop_3_sequential_1_batch_normalization_11_beta: P
Bassignvariableop_4_sequential_1_batch_normalization_11_moving_mean: T
Fassignvariableop_5_sequential_1_batch_normalization_11_moving_variance: B
/assignvariableop_6_sequential_1_dense_13_kernel:	 ?<
-assignvariableop_7_sequential_1_dense_13_bias:	?K
<assignvariableop_8_sequential_1_batch_normalization_10_gamma:	?J
;assignvariableop_9_sequential_1_batch_normalization_10_beta:	?R
Cassignvariableop_10_sequential_1_batch_normalization_10_moving_mean:	?V
Gassignvariableop_11_sequential_1_batch_normalization_10_moving_variance:	?C
0assignvariableop_12_sequential_1_dense_12_kernel:	?@<
.assignvariableop_13_sequential_1_dense_12_bias:@J
<assignvariableop_14_sequential_1_batch_normalization_9_gamma:@I
;assignvariableop_15_sequential_1_batch_normalization_9_beta:@P
Bassignvariableop_16_sequential_1_batch_normalization_9_moving_mean:@T
Fassignvariableop_17_sequential_1_batch_normalization_9_moving_variance:@C
0assignvariableop_18_sequential_1_dense_11_kernel:	@?=
.assignvariableop_19_sequential_1_dense_11_bias:	?K
<assignvariableop_20_sequential_1_batch_normalization_8_gamma:	?J
;assignvariableop_21_sequential_1_batch_normalization_8_beta:	?Q
Bassignvariableop_22_sequential_1_batch_normalization_8_moving_mean:	?U
Fassignvariableop_23_sequential_1_batch_normalization_8_moving_variance:	?C
0assignvariableop_24_sequential_1_dense_10_kernel:	?	<
.assignvariableop_25_sequential_1_dense_10_bias:	'
assignvariableop_26_adam_iter:	 )
assignvariableop_27_adam_beta_1: )
assignvariableop_28_adam_beta_2: (
assignvariableop_29_adam_decay: 0
&assignvariableop_30_adam_learning_rate: #
assignvariableop_31_total: #
assignvariableop_32_count: %
assignvariableop_33_total_1: %
assignvariableop_34_count_1: I
7assignvariableop_35_adam_sequential_1_dense_14_kernel_m: C
5assignvariableop_36_adam_sequential_1_dense_14_bias_m: R
Dassignvariableop_37_adam_sequential_1_batch_normalization_11_gamma_m: Q
Cassignvariableop_38_adam_sequential_1_batch_normalization_11_beta_m: J
7assignvariableop_39_adam_sequential_1_dense_13_kernel_m:	 ?D
5assignvariableop_40_adam_sequential_1_dense_13_bias_m:	?S
Dassignvariableop_41_adam_sequential_1_batch_normalization_10_gamma_m:	?R
Cassignvariableop_42_adam_sequential_1_batch_normalization_10_beta_m:	?J
7assignvariableop_43_adam_sequential_1_dense_12_kernel_m:	?@C
5assignvariableop_44_adam_sequential_1_dense_12_bias_m:@Q
Cassignvariableop_45_adam_sequential_1_batch_normalization_9_gamma_m:@P
Bassignvariableop_46_adam_sequential_1_batch_normalization_9_beta_m:@J
7assignvariableop_47_adam_sequential_1_dense_11_kernel_m:	@?D
5assignvariableop_48_adam_sequential_1_dense_11_bias_m:	?R
Cassignvariableop_49_adam_sequential_1_batch_normalization_8_gamma_m:	?Q
Bassignvariableop_50_adam_sequential_1_batch_normalization_8_beta_m:	?J
7assignvariableop_51_adam_sequential_1_dense_10_kernel_m:	?	C
5assignvariableop_52_adam_sequential_1_dense_10_bias_m:	I
7assignvariableop_53_adam_sequential_1_dense_14_kernel_v: C
5assignvariableop_54_adam_sequential_1_dense_14_bias_v: R
Dassignvariableop_55_adam_sequential_1_batch_normalization_11_gamma_v: Q
Cassignvariableop_56_adam_sequential_1_batch_normalization_11_beta_v: J
7assignvariableop_57_adam_sequential_1_dense_13_kernel_v:	 ?D
5assignvariableop_58_adam_sequential_1_dense_13_bias_v:	?S
Dassignvariableop_59_adam_sequential_1_batch_normalization_10_gamma_v:	?R
Cassignvariableop_60_adam_sequential_1_batch_normalization_10_beta_v:	?J
7assignvariableop_61_adam_sequential_1_dense_12_kernel_v:	?@C
5assignvariableop_62_adam_sequential_1_dense_12_bias_v:@Q
Cassignvariableop_63_adam_sequential_1_batch_normalization_9_gamma_v:@P
Bassignvariableop_64_adam_sequential_1_batch_normalization_9_beta_v:@J
7assignvariableop_65_adam_sequential_1_dense_11_kernel_v:	@?D
5assignvariableop_66_adam_sequential_1_dense_11_bias_v:	?R
Cassignvariableop_67_adam_sequential_1_batch_normalization_8_gamma_v:	?Q
Bassignvariableop_68_adam_sequential_1_batch_normalization_8_beta_v:	?J
7assignvariableop_69_adam_sequential_1_dense_10_kernel_v:	?	C
5assignvariableop_70_adam_sequential_1_dense_10_bias_v:	
identity_72??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_8?AssignVariableOp_9?'
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:H*
dtype0*?&
value?&B?&HB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:H*
dtype0*?
value?B?HB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*V
dtypesL
J2H	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp-assignvariableop_sequential_1_dense_14_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp-assignvariableop_1_sequential_1_dense_14_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp<assignvariableop_2_sequential_1_batch_normalization_11_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp;assignvariableop_3_sequential_1_batch_normalization_11_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpBassignvariableop_4_sequential_1_batch_normalization_11_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpFassignvariableop_5_sequential_1_batch_normalization_11_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp/assignvariableop_6_sequential_1_dense_13_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp-assignvariableop_7_sequential_1_dense_13_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp<assignvariableop_8_sequential_1_batch_normalization_10_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp;assignvariableop_9_sequential_1_batch_normalization_10_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpCassignvariableop_10_sequential_1_batch_normalization_10_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpGassignvariableop_11_sequential_1_batch_normalization_10_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp0assignvariableop_12_sequential_1_dense_12_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp.assignvariableop_13_sequential_1_dense_12_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp<assignvariableop_14_sequential_1_batch_normalization_9_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp;assignvariableop_15_sequential_1_batch_normalization_9_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpBassignvariableop_16_sequential_1_batch_normalization_9_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpFassignvariableop_17_sequential_1_batch_normalization_9_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp0assignvariableop_18_sequential_1_dense_11_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp.assignvariableop_19_sequential_1_dense_11_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp<assignvariableop_20_sequential_1_batch_normalization_8_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp;assignvariableop_21_sequential_1_batch_normalization_8_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpBassignvariableop_22_sequential_1_batch_normalization_8_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpFassignvariableop_23_sequential_1_batch_normalization_8_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp0assignvariableop_24_sequential_1_dense_10_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp.assignvariableop_25_sequential_1_dense_10_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpassignvariableop_26_adam_iterIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpassignvariableop_27_adam_beta_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpassignvariableop_28_adam_beta_2Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpassignvariableop_29_adam_decayIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_learning_rateIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpassignvariableop_31_totalIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOpassignvariableop_32_countIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOpassignvariableop_33_total_1Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOpassignvariableop_34_count_1Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp7assignvariableop_35_adam_sequential_1_dense_14_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp5assignvariableop_36_adam_sequential_1_dense_14_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOpDassignvariableop_37_adam_sequential_1_batch_normalization_11_gamma_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOpCassignvariableop_38_adam_sequential_1_batch_normalization_11_beta_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp7assignvariableop_39_adam_sequential_1_dense_13_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp5assignvariableop_40_adam_sequential_1_dense_13_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOpDassignvariableop_41_adam_sequential_1_batch_normalization_10_gamma_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOpCassignvariableop_42_adam_sequential_1_batch_normalization_10_beta_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp7assignvariableop_43_adam_sequential_1_dense_12_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp5assignvariableop_44_adam_sequential_1_dense_12_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOpCassignvariableop_45_adam_sequential_1_batch_normalization_9_gamma_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOpBassignvariableop_46_adam_sequential_1_batch_normalization_9_beta_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp7assignvariableop_47_adam_sequential_1_dense_11_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp5assignvariableop_48_adam_sequential_1_dense_11_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOpCassignvariableop_49_adam_sequential_1_batch_normalization_8_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOpBassignvariableop_50_adam_sequential_1_batch_normalization_8_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp7assignvariableop_51_adam_sequential_1_dense_10_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp5assignvariableop_52_adam_sequential_1_dense_10_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp7assignvariableop_53_adam_sequential_1_dense_14_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp5assignvariableop_54_adam_sequential_1_dense_14_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOpDassignvariableop_55_adam_sequential_1_batch_normalization_11_gamma_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOpCassignvariableop_56_adam_sequential_1_batch_normalization_11_beta_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp7assignvariableop_57_adam_sequential_1_dense_13_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp5assignvariableop_58_adam_sequential_1_dense_13_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOpDassignvariableop_59_adam_sequential_1_batch_normalization_10_gamma_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOpCassignvariableop_60_adam_sequential_1_batch_normalization_10_beta_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp7assignvariableop_61_adam_sequential_1_dense_12_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp5assignvariableop_62_adam_sequential_1_dense_12_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOpCassignvariableop_63_adam_sequential_1_batch_normalization_9_gamma_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOpBassignvariableop_64_adam_sequential_1_batch_normalization_9_beta_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp7assignvariableop_65_adam_sequential_1_dense_11_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp5assignvariableop_66_adam_sequential_1_dense_11_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOpCassignvariableop_67_adam_sequential_1_batch_normalization_8_gamma_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOpBassignvariableop_68_adam_sequential_1_batch_normalization_8_beta_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp7assignvariableop_69_adam_sequential_1_dense_10_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp5assignvariableop_70_adam_sequential_1_dense_10_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_709
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_71Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_71?
Identity_72IdentityIdentity_71:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_72"#
identity_72Identity_72:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_560645

inputs+
cast_readvariableop_resource:	?-
cast_1_readvariableop_resource:	?-
cast_2_readvariableop_resource:	?-
cast_3_readvariableop_resource:	?
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
D__inference_dense_13_layer_call_and_return_conditional_losses_563996

inputs1
matmul_readvariableop_resource:	 ?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02@
>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp?
/sequential_1/dense_13/kernel/Regularizer/SquareSquareFsequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 ?21
/sequential_1/dense_13/kernel/Regularizer/Square?
.sequential_1/dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       20
.sequential_1/dense_13/kernel/Regularizer/Const?
,sequential_1/dense_13/kernel/Regularizer/SumSum3sequential_1/dense_13/kernel/Regularizer/Square:y:07sequential_1/dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2.
,sequential_1/dense_13/kernel/Regularizer/Sum?
.sequential_1/dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_1/dense_13/kernel/Regularizer/mul/x?
,sequential_1/dense_13/kernel/Regularizer/mulMul7sequential_1/dense_13/kernel/Regularizer/mul/x:output:05sequential_1/dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,sequential_1/dense_13/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp?^sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2?
>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
__inference_loss_fn_1_564330Z
Gsequential_1_dense_12_kernel_regularizer_square_readvariableop_resource:	?@
identity??>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp?
>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpGsequential_1_dense_12_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	?@*
dtype02@
>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp?
/sequential_1/dense_12/kernel/Regularizer/SquareSquareFsequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?@21
/sequential_1/dense_12/kernel/Regularizer/Square?
.sequential_1/dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       20
.sequential_1/dense_12/kernel/Regularizer/Const?
,sequential_1/dense_12/kernel/Regularizer/SumSum3sequential_1/dense_12/kernel/Regularizer/Square:y:07sequential_1/dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2.
,sequential_1/dense_12/kernel/Regularizer/Sum?
.sequential_1/dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>20
.sequential_1/dense_12/kernel/Regularizer/mul/x?
,sequential_1/dense_12/kernel/Regularizer/mulMul7sequential_1/dense_12/kernel/Regularizer/mul/x:output:05sequential_1/dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,sequential_1/dense_12/kernel/Regularizer/mul?
IdentityIdentity0sequential_1/dense_12/kernel/Regularizer/mul:z:0?^sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2?
>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp
?&
?	
-__inference_sequential_1_layer_call_fn_562227
category_embed_1
category_embed_2
category_embed_3
category_embed_4
category_embed_5
city_embed_1
city_embed_2
city_embed_3
city_embed_4
city_embed_5	
colon

commas
dash

exclam	
money	
month
parenthesis
state_embed_1
state_embed_2
state_embed_3
state_embed_4
state_embed_5
weekday
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:	 ?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:	?@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:	@?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?

unknown_23:	?	

unknown_24:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallcategory_embed_1category_embed_2category_embed_3category_embed_4category_embed_5city_embed_1city_embed_2city_embed_3city_embed_4city_embed_5coloncommasdashexclammoneymonthparenthesisstate_embed_1state_embed_2state_embed_3state_embed_4state_embed_5weekdayunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*4
_read_only_resource_inputs
!"#$'()*-./0*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_5620932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
'
_output_shapes
:?????????
*
_user_specified_namecategory_embed_1:YU
'
_output_shapes
:?????????
*
_user_specified_namecategory_embed_2:YU
'
_output_shapes
:?????????
*
_user_specified_namecategory_embed_3:YU
'
_output_shapes
:?????????
*
_user_specified_namecategory_embed_4:YU
'
_output_shapes
:?????????
*
_user_specified_namecategory_embed_5:UQ
'
_output_shapes
:?????????
&
_user_specified_namecity_embed_1:UQ
'
_output_shapes
:?????????
&
_user_specified_namecity_embed_2:UQ
'
_output_shapes
:?????????
&
_user_specified_namecity_embed_3:UQ
'
_output_shapes
:?????????
&
_user_specified_namecity_embed_4:U	Q
'
_output_shapes
:?????????
&
_user_specified_namecity_embed_5:N
J
'
_output_shapes
:?????????

_user_specified_namecolon:OK
'
_output_shapes
:?????????
 
_user_specified_namecommas:MI
'
_output_shapes
:?????????

_user_specified_namedash:OK
'
_output_shapes
:?????????
 
_user_specified_nameexclam:NJ
'
_output_shapes
:?????????

_user_specified_namemoney:NJ
'
_output_shapes
:?????????

_user_specified_namemonth:TP
'
_output_shapes
:?????????
%
_user_specified_nameparenthesis:VR
'
_output_shapes
:?????????
'
_user_specified_namestate_embed_1:VR
'
_output_shapes
:?????????
'
_user_specified_namestate_embed_2:VR
'
_output_shapes
:?????????
'
_user_specified_namestate_embed_3:VR
'
_output_shapes
:?????????
'
_user_specified_namestate_embed_4:VR
'
_output_shapes
:?????????
'
_user_specified_namestate_embed_5:PL
'
_output_shapes
:?????????
!
_user_specified_name	weekday
?

?
D__inference_dense_11_layer_call_and_return_conditional_losses_564208

inputs1
matmul_readvariableop_resource:	@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_11_layer_call_fn_563897

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_5604832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?)
?
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_564076

inputs6
'assignmovingavg_readvariableop_resource:	?8
)assignmovingavg_1_readvariableop_resource:	?+
cast_readvariableop_resource:	?-
cast_1_readvariableop_resource:	?
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_10_layer_call_fn_564022

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_5607052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_564319Z
Gsequential_1_dense_13_kernel_regularizer_square_readvariableop_resource:	 ?
identity??>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp?
>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpGsequential_1_dense_13_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	 ?*
dtype02@
>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp?
/sequential_1/dense_13/kernel/Regularizer/SquareSquareFsequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 ?21
/sequential_1/dense_13/kernel/Regularizer/Square?
.sequential_1/dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       20
.sequential_1/dense_13/kernel/Regularizer/Const?
,sequential_1/dense_13/kernel/Regularizer/SumSum3sequential_1/dense_13/kernel/Regularizer/Square:y:07sequential_1/dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2.
,sequential_1/dense_13/kernel/Regularizer/Sum?
.sequential_1/dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_1/dense_13/kernel/Regularizer/mul/x?
,sequential_1/dense_13/kernel/Regularizer/mulMul7sequential_1/dense_13/kernel/Regularizer/mul/x:output:05sequential_1/dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,sequential_1/dense_13/kernel/Regularizer/mul?
IdentityIdentity0sequential_1/dense_13/kernel/Regularizer/mul:z:0?^sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2?
>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp
?
?
D__inference_dense_12_layer_call_and_return_conditional_losses_561447

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02@
>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp?
/sequential_1/dense_12/kernel/Regularizer/SquareSquareFsequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?@21
/sequential_1/dense_12/kernel/Regularizer/Square?
.sequential_1/dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       20
.sequential_1/dense_12/kernel/Regularizer/Const?
,sequential_1/dense_12/kernel/Regularizer/SumSum3sequential_1/dense_12/kernel/Regularizer/Square:y:07sequential_1/dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2.
,sequential_1/dense_12/kernel/Regularizer/Sum?
.sequential_1/dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>20
.sequential_1/dense_12/kernel/Regularizer/mul/x?
,sequential_1/dense_12/kernel/Regularizer/mulMul7sequential_1/dense_12/kernel/Regularizer/mul/x:output:05sequential_1/dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,sequential_1/dense_12/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp?^sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2?
>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
D__inference_dense_10_layer_call_and_return_conditional_losses_564308

inputs1
matmul_readvariableop_resource:	?	-
biasadd_readvariableop_resource:	
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????	2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?l
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_561518

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13
	inputs_14
	inputs_15
	inputs_16
	inputs_17
	inputs_18
	inputs_19
	inputs_20
	inputs_21
	inputs_22!
dense_14_561384: 
dense_14_561386: +
batch_normalization_11_561389: +
batch_normalization_11_561391: +
batch_normalization_11_561393: +
batch_normalization_11_561395: "
dense_13_561416:	 ?
dense_13_561418:	?,
batch_normalization_10_561421:	?,
batch_normalization_10_561423:	?,
batch_normalization_10_561425:	?,
batch_normalization_10_561427:	?"
dense_12_561448:	?@
dense_12_561450:@*
batch_normalization_9_561453:@*
batch_normalization_9_561455:@*
batch_normalization_9_561457:@*
batch_normalization_9_561459:@"
dense_11_561474:	@?
dense_11_561476:	?+
batch_normalization_8_561479:	?+
batch_normalization_8_561481:	?+
batch_normalization_8_561483:	?+
batch_normalization_8_561485:	?"
dense_10_561500:	?	
dense_10_561502:	
identity??.batch_normalization_10/StatefulPartitionedCall?.batch_normalization_11/StatefulPartitionedCall?-batch_normalization_8/StatefulPartitionedCall?-batch_normalization_9/StatefulPartitionedCall? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall? dense_14/StatefulPartitionedCall?>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp?>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp?
 dense_features_1/PartitionedCallPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12	inputs_13	inputs_14	inputs_15	inputs_16	inputs_17	inputs_18	inputs_19	inputs_20	inputs_21	inputs_22*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_dense_features_1_layer_call_and_return_conditional_losses_5613702"
 dense_features_1/PartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_features_1/PartitionedCall:output:0dense_14_561384dense_14_561386*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_5613832"
 dense_14/StatefulPartitionedCall?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0batch_normalization_11_561389batch_normalization_11_561391batch_normalization_11_561393batch_normalization_11_561395*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_56048320
.batch_normalization_11/StatefulPartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0dense_13_561416dense_13_561418*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_5614152"
 dense_13/StatefulPartitionedCall?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0batch_normalization_10_561421batch_normalization_10_561423batch_normalization_10_561425batch_normalization_10_561427*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_56064520
.batch_normalization_10/StatefulPartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0dense_12_561448dense_12_561450*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_5614472"
 dense_12/StatefulPartitionedCall?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0batch_normalization_9_561453batch_normalization_9_561455batch_normalization_9_561457batch_normalization_9_561459*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_5608072/
-batch_normalization_9/StatefulPartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0dense_11_561474dense_11_561476*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_5614732"
 dense_11/StatefulPartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0batch_normalization_8_561479batch_normalization_8_561481batch_normalization_8_561483batch_normalization_8_561485*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_5609692/
-batch_normalization_8/StatefulPartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0dense_10_561500dense_10_561502*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_5614992"
 dense_10/StatefulPartitionedCall?
>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_13_561416*
_output_shapes
:	 ?*
dtype02@
>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp?
/sequential_1/dense_13/kernel/Regularizer/SquareSquareFsequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 ?21
/sequential_1/dense_13/kernel/Regularizer/Square?
.sequential_1/dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       20
.sequential_1/dense_13/kernel/Regularizer/Const?
,sequential_1/dense_13/kernel/Regularizer/SumSum3sequential_1/dense_13/kernel/Regularizer/Square:y:07sequential_1/dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2.
,sequential_1/dense_13/kernel/Regularizer/Sum?
.sequential_1/dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_1/dense_13/kernel/Regularizer/mul/x?
,sequential_1/dense_13/kernel/Regularizer/mulMul7sequential_1/dense_13/kernel/Regularizer/mul/x:output:05sequential_1/dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,sequential_1/dense_13/kernel/Regularizer/mul?
>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_12_561448*
_output_shapes
:	?@*
dtype02@
>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp?
/sequential_1/dense_12/kernel/Regularizer/SquareSquareFsequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?@21
/sequential_1/dense_12/kernel/Regularizer/Square?
.sequential_1/dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       20
.sequential_1/dense_12/kernel/Regularizer/Const?
,sequential_1/dense_12/kernel/Regularizer/SumSum3sequential_1/dense_12/kernel/Regularizer/Square:y:07sequential_1/dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2.
,sequential_1/dense_12/kernel/Regularizer/Sum?
.sequential_1/dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>20
.sequential_1/dense_12/kernel/Regularizer/mul/x?
,sequential_1/dense_12/kernel/Regularizer/mulMul7sequential_1/dense_12/kernel/Regularizer/mul/x:output:05sequential_1/dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,sequential_1/dense_12/kernel/Regularizer/mul?
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall?^sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp?^sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2?
>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp2?
>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O	K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O
K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_10_layer_call_fn_564009

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_5606452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
D__inference_dense_10_layer_call_and_return_conditional_losses_561499

inputs1
matmul_readvariableop_resource:	?	-
biasadd_readvariableop_resource:	
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????	2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_564154

inputs*
cast_readvariableop_resource:@,
cast_1_readvariableop_resource:@,
cast_2_readvariableop_resource:@,
cast_3_readvariableop_resource:@
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:@*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:@*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:@*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:@*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????@2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????@2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
)__inference_dense_14_layer_call_fn_563873

inputs
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_5613832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?"
?
1__inference_dense_features_1_layer_call_fn_563413
features_category_embed_1
features_category_embed_2
features_category_embed_3
features_category_embed_4
features_category_embed_5
features_city_embed_1
features_city_embed_2
features_city_embed_3
features_city_embed_4
features_city_embed_5
features_colon
features_commas
features_dash
features_exclam
features_money
features_month
features_parenthesis
features_state_embed_1
features_state_embed_2
features_state_embed_3
features_state_embed_4
features_state_embed_5
features_weekday
identity?
PartitionedCallPartitionedCallfeatures_category_embed_1features_category_embed_2features_category_embed_3features_category_embed_4features_category_embed_5features_city_embed_1features_city_embed_2features_city_embed_3features_city_embed_4features_city_embed_5features_colonfeatures_commasfeatures_dashfeatures_exclamfeatures_moneyfeatures_monthfeatures_parenthesisfeatures_state_embed_1features_state_embed_2features_state_embed_3features_state_embed_4features_state_embed_5features_weekday*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_dense_features_1_layer_call_and_return_conditional_losses_5613702
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:b ^
'
_output_shapes
:?????????
3
_user_specified_namefeatures/category_embed_1:b^
'
_output_shapes
:?????????
3
_user_specified_namefeatures/category_embed_2:b^
'
_output_shapes
:?????????
3
_user_specified_namefeatures/category_embed_3:b^
'
_output_shapes
:?????????
3
_user_specified_namefeatures/category_embed_4:b^
'
_output_shapes
:?????????
3
_user_specified_namefeatures/category_embed_5:^Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/city_embed_1:^Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/city_embed_2:^Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/city_embed_3:^Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/city_embed_4:^	Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/city_embed_5:W
S
'
_output_shapes
:?????????
(
_user_specified_namefeatures/colon:XT
'
_output_shapes
:?????????
)
_user_specified_namefeatures/commas:VR
'
_output_shapes
:?????????
'
_user_specified_namefeatures/dash:XT
'
_output_shapes
:?????????
)
_user_specified_namefeatures/exclam:WS
'
_output_shapes
:?????????
(
_user_specified_namefeatures/money:WS
'
_output_shapes
:?????????
(
_user_specified_namefeatures/month:]Y
'
_output_shapes
:?????????
.
_user_specified_namefeatures/parenthesis:_[
'
_output_shapes
:?????????
0
_user_specified_namefeatures/state_embed_1:_[
'
_output_shapes
:?????????
0
_user_specified_namefeatures/state_embed_2:_[
'
_output_shapes
:?????????
0
_user_specified_namefeatures/state_embed_3:_[
'
_output_shapes
:?????????
0
_user_specified_namefeatures/state_embed_4:_[
'
_output_shapes
:?????????
0
_user_specified_namefeatures/state_embed_5:YU
'
_output_shapes
:?????????
*
_user_specified_namefeatures/weekday
?)
?
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_563964

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: *
cast_readvariableop_resource: ,
cast_1_readvariableop_resource: 
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

: 2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:????????? 2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
: *
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
: *
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:????????? 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:????????? 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
)__inference_dense_13_layer_call_fn_563979

inputs
unknown:	 ?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_5614152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_560969

inputs+
cast_readvariableop_resource:	?-
cast_1_readvariableop_resource:	?-
cast_2_readvariableop_resource:	?-
cast_3_readvariableop_resource:	?
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?*
?

-__inference_sequential_1_layer_call_fn_562684
inputs_category_embed_1
inputs_category_embed_2
inputs_category_embed_3
inputs_category_embed_4
inputs_category_embed_5
inputs_city_embed_1
inputs_city_embed_2
inputs_city_embed_3
inputs_city_embed_4
inputs_city_embed_5
inputs_colon
inputs_commas
inputs_dash
inputs_exclam
inputs_money
inputs_month
inputs_parenthesis
inputs_state_embed_1
inputs_state_embed_2
inputs_state_embed_3
inputs_state_embed_4
inputs_state_embed_5
inputs_weekday
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:	 ?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:	?@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:	@?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?

unknown_23:	?	

unknown_24:	
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallinputs_category_embed_1inputs_category_embed_2inputs_category_embed_3inputs_category_embed_4inputs_category_embed_5inputs_city_embed_1inputs_city_embed_2inputs_city_embed_3inputs_city_embed_4inputs_city_embed_5inputs_coloninputs_commasinputs_dashinputs_exclaminputs_moneyinputs_monthinputs_parenthesisinputs_state_embed_1inputs_state_embed_2inputs_state_embed_3inputs_state_embed_4inputs_state_embed_5inputs_weekdayunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*4
_read_only_resource_inputs
!"#$'()*-./0*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_5620932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
'
_output_shapes
:?????????
1
_user_specified_nameinputs/category_embed_1:`\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/category_embed_2:`\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/category_embed_3:`\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/category_embed_4:`\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/category_embed_5:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/city_embed_1:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/city_embed_2:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/city_embed_3:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/city_embed_4:\	X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/city_embed_5:U
Q
'
_output_shapes
:?????????
&
_user_specified_nameinputs/colon:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/commas:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/dash:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/exclam:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/money:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/month:[W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/parenthesis:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/state_embed_1:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/state_embed_2:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/state_embed_3:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/state_embed_4:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/state_embed_5:WS
'
_output_shapes
:?????????
(
_user_specified_nameinputs/weekday
?
?
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_560807

inputs*
cast_readvariableop_resource:@,
cast_1_readvariableop_resource:@,
cast_2_readvariableop_resource:@,
cast_3_readvariableop_resource:@
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:@*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:@*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:@*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:@*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????@2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????@2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?
L__inference_dense_features_1_layer_call_and_return_conditional_losses_561370
features

features_1

features_2

features_3

features_4

features_5

features_6

features_7

features_8

features_9
features_10
features_11
features_12
features_13
features_14
features_15
features_16
features_17
features_18
features_19
features_20
features_21
features_22
identityh
category_embed_1/ShapeShapefeatures*
T0*
_output_shapes
:2
category_embed_1/Shape?
$category_embed_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$category_embed_1/strided_slice/stack?
&category_embed_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_1/strided_slice/stack_1?
&category_embed_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_1/strided_slice/stack_2?
category_embed_1/strided_sliceStridedSlicecategory_embed_1/Shape:output:0-category_embed_1/strided_slice/stack:output:0/category_embed_1/strided_slice/stack_1:output:0/category_embed_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
category_embed_1/strided_slice?
 category_embed_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 category_embed_1/Reshape/shape/1?
category_embed_1/Reshape/shapePack'category_embed_1/strided_slice:output:0)category_embed_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2 
category_embed_1/Reshape/shape?
category_embed_1/ReshapeReshapefeatures'category_embed_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
category_embed_1/Reshapej
category_embed_2/ShapeShape
features_1*
T0*
_output_shapes
:2
category_embed_2/Shape?
$category_embed_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$category_embed_2/strided_slice/stack?
&category_embed_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_2/strided_slice/stack_1?
&category_embed_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_2/strided_slice/stack_2?
category_embed_2/strided_sliceStridedSlicecategory_embed_2/Shape:output:0-category_embed_2/strided_slice/stack:output:0/category_embed_2/strided_slice/stack_1:output:0/category_embed_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
category_embed_2/strided_slice?
 category_embed_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 category_embed_2/Reshape/shape/1?
category_embed_2/Reshape/shapePack'category_embed_2/strided_slice:output:0)category_embed_2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2 
category_embed_2/Reshape/shape?
category_embed_2/ReshapeReshape
features_1'category_embed_2/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
category_embed_2/Reshapej
category_embed_3/ShapeShape
features_2*
T0*
_output_shapes
:2
category_embed_3/Shape?
$category_embed_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$category_embed_3/strided_slice/stack?
&category_embed_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_3/strided_slice/stack_1?
&category_embed_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_3/strided_slice/stack_2?
category_embed_3/strided_sliceStridedSlicecategory_embed_3/Shape:output:0-category_embed_3/strided_slice/stack:output:0/category_embed_3/strided_slice/stack_1:output:0/category_embed_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
category_embed_3/strided_slice?
 category_embed_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 category_embed_3/Reshape/shape/1?
category_embed_3/Reshape/shapePack'category_embed_3/strided_slice:output:0)category_embed_3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2 
category_embed_3/Reshape/shape?
category_embed_3/ReshapeReshape
features_2'category_embed_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
category_embed_3/Reshapej
category_embed_4/ShapeShape
features_3*
T0*
_output_shapes
:2
category_embed_4/Shape?
$category_embed_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$category_embed_4/strided_slice/stack?
&category_embed_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_4/strided_slice/stack_1?
&category_embed_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_4/strided_slice/stack_2?
category_embed_4/strided_sliceStridedSlicecategory_embed_4/Shape:output:0-category_embed_4/strided_slice/stack:output:0/category_embed_4/strided_slice/stack_1:output:0/category_embed_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
category_embed_4/strided_slice?
 category_embed_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 category_embed_4/Reshape/shape/1?
category_embed_4/Reshape/shapePack'category_embed_4/strided_slice:output:0)category_embed_4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2 
category_embed_4/Reshape/shape?
category_embed_4/ReshapeReshape
features_3'category_embed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
category_embed_4/Reshapej
category_embed_5/ShapeShape
features_4*
T0*
_output_shapes
:2
category_embed_5/Shape?
$category_embed_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$category_embed_5/strided_slice/stack?
&category_embed_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_5/strided_slice/stack_1?
&category_embed_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_5/strided_slice/stack_2?
category_embed_5/strided_sliceStridedSlicecategory_embed_5/Shape:output:0-category_embed_5/strided_slice/stack:output:0/category_embed_5/strided_slice/stack_1:output:0/category_embed_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
category_embed_5/strided_slice?
 category_embed_5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 category_embed_5/Reshape/shape/1?
category_embed_5/Reshape/shapePack'category_embed_5/strided_slice:output:0)category_embed_5/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2 
category_embed_5/Reshape/shape?
category_embed_5/ReshapeReshape
features_4'category_embed_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
category_embed_5/Reshapeb
city_embed_1/ShapeShape
features_5*
T0*
_output_shapes
:2
city_embed_1/Shape?
 city_embed_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 city_embed_1/strided_slice/stack?
"city_embed_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_1/strided_slice/stack_1?
"city_embed_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_1/strided_slice/stack_2?
city_embed_1/strided_sliceStridedSlicecity_embed_1/Shape:output:0)city_embed_1/strided_slice/stack:output:0+city_embed_1/strided_slice/stack_1:output:0+city_embed_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
city_embed_1/strided_slice~
city_embed_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
city_embed_1/Reshape/shape/1?
city_embed_1/Reshape/shapePack#city_embed_1/strided_slice:output:0%city_embed_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
city_embed_1/Reshape/shape?
city_embed_1/ReshapeReshape
features_5#city_embed_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
city_embed_1/Reshapeb
city_embed_2/ShapeShape
features_6*
T0*
_output_shapes
:2
city_embed_2/Shape?
 city_embed_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 city_embed_2/strided_slice/stack?
"city_embed_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_2/strided_slice/stack_1?
"city_embed_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_2/strided_slice/stack_2?
city_embed_2/strided_sliceStridedSlicecity_embed_2/Shape:output:0)city_embed_2/strided_slice/stack:output:0+city_embed_2/strided_slice/stack_1:output:0+city_embed_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
city_embed_2/strided_slice~
city_embed_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
city_embed_2/Reshape/shape/1?
city_embed_2/Reshape/shapePack#city_embed_2/strided_slice:output:0%city_embed_2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
city_embed_2/Reshape/shape?
city_embed_2/ReshapeReshape
features_6#city_embed_2/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
city_embed_2/Reshapeb
city_embed_3/ShapeShape
features_7*
T0*
_output_shapes
:2
city_embed_3/Shape?
 city_embed_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 city_embed_3/strided_slice/stack?
"city_embed_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_3/strided_slice/stack_1?
"city_embed_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_3/strided_slice/stack_2?
city_embed_3/strided_sliceStridedSlicecity_embed_3/Shape:output:0)city_embed_3/strided_slice/stack:output:0+city_embed_3/strided_slice/stack_1:output:0+city_embed_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
city_embed_3/strided_slice~
city_embed_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
city_embed_3/Reshape/shape/1?
city_embed_3/Reshape/shapePack#city_embed_3/strided_slice:output:0%city_embed_3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
city_embed_3/Reshape/shape?
city_embed_3/ReshapeReshape
features_7#city_embed_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
city_embed_3/Reshapeb
city_embed_4/ShapeShape
features_8*
T0*
_output_shapes
:2
city_embed_4/Shape?
 city_embed_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 city_embed_4/strided_slice/stack?
"city_embed_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_4/strided_slice/stack_1?
"city_embed_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_4/strided_slice/stack_2?
city_embed_4/strided_sliceStridedSlicecity_embed_4/Shape:output:0)city_embed_4/strided_slice/stack:output:0+city_embed_4/strided_slice/stack_1:output:0+city_embed_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
city_embed_4/strided_slice~
city_embed_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
city_embed_4/Reshape/shape/1?
city_embed_4/Reshape/shapePack#city_embed_4/strided_slice:output:0%city_embed_4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
city_embed_4/Reshape/shape?
city_embed_4/ReshapeReshape
features_8#city_embed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
city_embed_4/Reshapeb
city_embed_5/ShapeShape
features_9*
T0*
_output_shapes
:2
city_embed_5/Shape?
 city_embed_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 city_embed_5/strided_slice/stack?
"city_embed_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_5/strided_slice/stack_1?
"city_embed_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_5/strided_slice/stack_2?
city_embed_5/strided_sliceStridedSlicecity_embed_5/Shape:output:0)city_embed_5/strided_slice/stack:output:0+city_embed_5/strided_slice/stack_1:output:0+city_embed_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
city_embed_5/strided_slice~
city_embed_5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
city_embed_5/Reshape/shape/1?
city_embed_5/Reshape/shapePack#city_embed_5/strided_slice:output:0%city_embed_5/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
city_embed_5/Reshape/shape?
city_embed_5/ReshapeReshape
features_9#city_embed_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
city_embed_5/ReshapeU
colon/ShapeShapefeatures_10*
T0*
_output_shapes
:2
colon/Shape?
colon/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
colon/strided_slice/stack?
colon/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
colon/strided_slice/stack_1?
colon/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
colon/strided_slice/stack_2?
colon/strided_sliceStridedSlicecolon/Shape:output:0"colon/strided_slice/stack:output:0$colon/strided_slice/stack_1:output:0$colon/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
colon/strided_slicep
colon/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
colon/Reshape/shape/1?
colon/Reshape/shapePackcolon/strided_slice:output:0colon/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
colon/Reshape/shape?
colon/ReshapeReshapefeatures_10colon/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
colon/ReshapeW
commas/ShapeShapefeatures_11*
T0*
_output_shapes
:2
commas/Shape?
commas/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
commas/strided_slice/stack?
commas/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
commas/strided_slice/stack_1?
commas/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
commas/strided_slice/stack_2?
commas/strided_sliceStridedSlicecommas/Shape:output:0#commas/strided_slice/stack:output:0%commas/strided_slice/stack_1:output:0%commas/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
commas/strided_slicer
commas/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
commas/Reshape/shape/1?
commas/Reshape/shapePackcommas/strided_slice:output:0commas/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
commas/Reshape/shape?
commas/ReshapeReshapefeatures_11commas/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
commas/ReshapeS

dash/ShapeShapefeatures_12*
T0*
_output_shapes
:2

dash/Shape~
dash/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
dash/strided_slice/stack?
dash/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
dash/strided_slice/stack_1?
dash/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
dash/strided_slice/stack_2?
dash/strided_sliceStridedSlicedash/Shape:output:0!dash/strided_slice/stack:output:0#dash/strided_slice/stack_1:output:0#dash/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dash/strided_slicen
dash/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
dash/Reshape/shape/1?
dash/Reshape/shapePackdash/strided_slice:output:0dash/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
dash/Reshape/shape?
dash/ReshapeReshapefeatures_12dash/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
dash/ReshapeW
exclam/ShapeShapefeatures_13*
T0*
_output_shapes
:2
exclam/Shape?
exclam/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
exclam/strided_slice/stack?
exclam/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
exclam/strided_slice/stack_1?
exclam/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
exclam/strided_slice/stack_2?
exclam/strided_sliceStridedSliceexclam/Shape:output:0#exclam/strided_slice/stack:output:0%exclam/strided_slice/stack_1:output:0%exclam/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
exclam/strided_slicer
exclam/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
exclam/Reshape/shape/1?
exclam/Reshape/shapePackexclam/strided_slice:output:0exclam/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
exclam/Reshape/shape?
exclam/ReshapeReshapefeatures_13exclam/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
exclam/ReshapeU
money/ShapeShapefeatures_14*
T0*
_output_shapes
:2
money/Shape?
money/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
money/strided_slice/stack?
money/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
money/strided_slice/stack_1?
money/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
money/strided_slice/stack_2?
money/strided_sliceStridedSlicemoney/Shape:output:0"money/strided_slice/stack:output:0$money/strided_slice/stack_1:output:0$money/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
money/strided_slicep
money/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
money/Reshape/shape/1?
money/Reshape/shapePackmoney/strided_slice:output:0money/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
money/Reshape/shape?
money/ReshapeReshapefeatures_14money/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
money/ReshapeU
month/ShapeShapefeatures_15*
T0*
_output_shapes
:2
month/Shape?
month/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
month/strided_slice/stack?
month/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
month/strided_slice/stack_1?
month/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
month/strided_slice/stack_2?
month/strided_sliceStridedSlicemonth/Shape:output:0"month/strided_slice/stack:output:0$month/strided_slice/stack_1:output:0$month/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
month/strided_slicep
month/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
month/Reshape/shape/1?
month/Reshape/shapePackmonth/strided_slice:output:0month/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
month/Reshape/shape?
month/ReshapeReshapefeatures_15month/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
month/Reshapea
parenthesis/ShapeShapefeatures_16*
T0*
_output_shapes
:2
parenthesis/Shape?
parenthesis/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
parenthesis/strided_slice/stack?
!parenthesis/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!parenthesis/strided_slice/stack_1?
!parenthesis/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!parenthesis/strided_slice/stack_2?
parenthesis/strided_sliceStridedSliceparenthesis/Shape:output:0(parenthesis/strided_slice/stack:output:0*parenthesis/strided_slice/stack_1:output:0*parenthesis/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
parenthesis/strided_slice|
parenthesis/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
parenthesis/Reshape/shape/1?
parenthesis/Reshape/shapePack"parenthesis/strided_slice:output:0$parenthesis/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
parenthesis/Reshape/shape?
parenthesis/ReshapeReshapefeatures_16"parenthesis/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
parenthesis/Reshapee
state_embed_1/ShapeShapefeatures_17*
T0*
_output_shapes
:2
state_embed_1/Shape?
!state_embed_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!state_embed_1/strided_slice/stack?
#state_embed_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_1/strided_slice/stack_1?
#state_embed_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_1/strided_slice/stack_2?
state_embed_1/strided_sliceStridedSlicestate_embed_1/Shape:output:0*state_embed_1/strided_slice/stack:output:0,state_embed_1/strided_slice/stack_1:output:0,state_embed_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
state_embed_1/strided_slice?
state_embed_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
state_embed_1/Reshape/shape/1?
state_embed_1/Reshape/shapePack$state_embed_1/strided_slice:output:0&state_embed_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
state_embed_1/Reshape/shape?
state_embed_1/ReshapeReshapefeatures_17$state_embed_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
state_embed_1/Reshapee
state_embed_2/ShapeShapefeatures_18*
T0*
_output_shapes
:2
state_embed_2/Shape?
!state_embed_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!state_embed_2/strided_slice/stack?
#state_embed_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_2/strided_slice/stack_1?
#state_embed_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_2/strided_slice/stack_2?
state_embed_2/strided_sliceStridedSlicestate_embed_2/Shape:output:0*state_embed_2/strided_slice/stack:output:0,state_embed_2/strided_slice/stack_1:output:0,state_embed_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
state_embed_2/strided_slice?
state_embed_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
state_embed_2/Reshape/shape/1?
state_embed_2/Reshape/shapePack$state_embed_2/strided_slice:output:0&state_embed_2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
state_embed_2/Reshape/shape?
state_embed_2/ReshapeReshapefeatures_18$state_embed_2/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
state_embed_2/Reshapee
state_embed_3/ShapeShapefeatures_19*
T0*
_output_shapes
:2
state_embed_3/Shape?
!state_embed_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!state_embed_3/strided_slice/stack?
#state_embed_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_3/strided_slice/stack_1?
#state_embed_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_3/strided_slice/stack_2?
state_embed_3/strided_sliceStridedSlicestate_embed_3/Shape:output:0*state_embed_3/strided_slice/stack:output:0,state_embed_3/strided_slice/stack_1:output:0,state_embed_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
state_embed_3/strided_slice?
state_embed_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
state_embed_3/Reshape/shape/1?
state_embed_3/Reshape/shapePack$state_embed_3/strided_slice:output:0&state_embed_3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
state_embed_3/Reshape/shape?
state_embed_3/ReshapeReshapefeatures_19$state_embed_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
state_embed_3/Reshapee
state_embed_4/ShapeShapefeatures_20*
T0*
_output_shapes
:2
state_embed_4/Shape?
!state_embed_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!state_embed_4/strided_slice/stack?
#state_embed_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_4/strided_slice/stack_1?
#state_embed_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_4/strided_slice/stack_2?
state_embed_4/strided_sliceStridedSlicestate_embed_4/Shape:output:0*state_embed_4/strided_slice/stack:output:0,state_embed_4/strided_slice/stack_1:output:0,state_embed_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
state_embed_4/strided_slice?
state_embed_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
state_embed_4/Reshape/shape/1?
state_embed_4/Reshape/shapePack$state_embed_4/strided_slice:output:0&state_embed_4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
state_embed_4/Reshape/shape?
state_embed_4/ReshapeReshapefeatures_20$state_embed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
state_embed_4/Reshapee
state_embed_5/ShapeShapefeatures_21*
T0*
_output_shapes
:2
state_embed_5/Shape?
!state_embed_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!state_embed_5/strided_slice/stack?
#state_embed_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_5/strided_slice/stack_1?
#state_embed_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_5/strided_slice/stack_2?
state_embed_5/strided_sliceStridedSlicestate_embed_5/Shape:output:0*state_embed_5/strided_slice/stack:output:0,state_embed_5/strided_slice/stack_1:output:0,state_embed_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
state_embed_5/strided_slice?
state_embed_5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
state_embed_5/Reshape/shape/1?
state_embed_5/Reshape/shapePack$state_embed_5/strided_slice:output:0&state_embed_5/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
state_embed_5/Reshape/shape?
state_embed_5/ReshapeReshapefeatures_21$state_embed_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
state_embed_5/ReshapeY
weekday/ShapeShapefeatures_22*
T0*
_output_shapes
:2
weekday/Shape?
weekday/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
weekday/strided_slice/stack?
weekday/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
weekday/strided_slice/stack_1?
weekday/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
weekday/strided_slice/stack_2?
weekday/strided_sliceStridedSliceweekday/Shape:output:0$weekday/strided_slice/stack:output:0&weekday/strided_slice/stack_1:output:0&weekday/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
weekday/strided_slicet
weekday/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
weekday/Reshape/shape/1?
weekday/Reshape/shapePackweekday/strided_slice:output:0 weekday/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
weekday/Reshape/shape?
weekday/ReshapeReshapefeatures_22weekday/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
weekday/Reshapee
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2!category_embed_1/Reshape:output:0!category_embed_2/Reshape:output:0!category_embed_3/Reshape:output:0!category_embed_4/Reshape:output:0!category_embed_5/Reshape:output:0city_embed_1/Reshape:output:0city_embed_2/Reshape:output:0city_embed_3/Reshape:output:0city_embed_4/Reshape:output:0city_embed_5/Reshape:output:0colon/Reshape:output:0commas/Reshape:output:0dash/Reshape:output:0exclam/Reshape:output:0money/Reshape:output:0month/Reshape:output:0parenthesis/Reshape:output:0state_embed_1/Reshape:output:0state_embed_2/Reshape:output:0state_embed_3/Reshape:output:0state_embed_4/Reshape:output:0state_embed_5/Reshape:output:0weekday/Reshape:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:Q	M
'
_output_shapes
:?????????
"
_user_specified_name
features:Q
M
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features
?
?
6__inference_batch_normalization_8_layer_call_fn_564234

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_5610292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?m
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_562427
category_embed_1
category_embed_2
category_embed_3
category_embed_4
category_embed_5
city_embed_1
city_embed_2
city_embed_3
city_embed_4
city_embed_5	
colon

commas
dash

exclam	
money	
month
parenthesis
state_embed_1
state_embed_2
state_embed_3
state_embed_4
state_embed_5
weekday!
dense_14_562353: 
dense_14_562355: +
batch_normalization_11_562358: +
batch_normalization_11_562360: +
batch_normalization_11_562362: +
batch_normalization_11_562364: "
dense_13_562367:	 ?
dense_13_562369:	?,
batch_normalization_10_562372:	?,
batch_normalization_10_562374:	?,
batch_normalization_10_562376:	?,
batch_normalization_10_562378:	?"
dense_12_562381:	?@
dense_12_562383:@*
batch_normalization_9_562386:@*
batch_normalization_9_562388:@*
batch_normalization_9_562390:@*
batch_normalization_9_562392:@"
dense_11_562395:	@?
dense_11_562397:	?+
batch_normalization_8_562400:	?+
batch_normalization_8_562402:	?+
batch_normalization_8_562404:	?+
batch_normalization_8_562406:	?"
dense_10_562409:	?	
dense_10_562411:	
identity??.batch_normalization_10/StatefulPartitionedCall?.batch_normalization_11/StatefulPartitionedCall?-batch_normalization_8/StatefulPartitionedCall?-batch_normalization_9/StatefulPartitionedCall? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall? dense_14/StatefulPartitionedCall?>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp?>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp?
 dense_features_1/PartitionedCallPartitionedCallcategory_embed_1category_embed_2category_embed_3category_embed_4category_embed_5city_embed_1city_embed_2city_embed_3city_embed_4city_embed_5coloncommasdashexclammoneymonthparenthesisstate_embed_1state_embed_2state_embed_3state_embed_4state_embed_5weekday*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_dense_features_1_layer_call_and_return_conditional_losses_5618872"
 dense_features_1/PartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_features_1/PartitionedCall:output:0dense_14_562353dense_14_562355*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_5613832"
 dense_14/StatefulPartitionedCall?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0batch_normalization_11_562358batch_normalization_11_562360batch_normalization_11_562362batch_normalization_11_562364*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_56054320
.batch_normalization_11/StatefulPartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0dense_13_562367dense_13_562369*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_5614152"
 dense_13/StatefulPartitionedCall?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0batch_normalization_10_562372batch_normalization_10_562374batch_normalization_10_562376batch_normalization_10_562378*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_56070520
.batch_normalization_10/StatefulPartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0dense_12_562381dense_12_562383*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_12_layer_call_and_return_conditional_losses_5614472"
 dense_12/StatefulPartitionedCall?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0batch_normalization_9_562386batch_normalization_9_562388batch_normalization_9_562390batch_normalization_9_562392*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_5608672/
-batch_normalization_9/StatefulPartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0dense_11_562395dense_11_562397*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_5614732"
 dense_11/StatefulPartitionedCall?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0batch_normalization_8_562400batch_normalization_8_562402batch_normalization_8_562404batch_normalization_8_562406*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_5610292/
-batch_normalization_8/StatefulPartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0dense_10_562409dense_10_562411*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_5614992"
 dense_10/StatefulPartitionedCall?
>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_13_562367*
_output_shapes
:	 ?*
dtype02@
>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp?
/sequential_1/dense_13/kernel/Regularizer/SquareSquareFsequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 ?21
/sequential_1/dense_13/kernel/Regularizer/Square?
.sequential_1/dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       20
.sequential_1/dense_13/kernel/Regularizer/Const?
,sequential_1/dense_13/kernel/Regularizer/SumSum3sequential_1/dense_13/kernel/Regularizer/Square:y:07sequential_1/dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2.
,sequential_1/dense_13/kernel/Regularizer/Sum?
.sequential_1/dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_1/dense_13/kernel/Regularizer/mul/x?
,sequential_1/dense_13/kernel/Regularizer/mulMul7sequential_1/dense_13/kernel/Regularizer/mul/x:output:05sequential_1/dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,sequential_1/dense_13/kernel/Regularizer/mul?
>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_12_562381*
_output_shapes
:	?@*
dtype02@
>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp?
/sequential_1/dense_12/kernel/Regularizer/SquareSquareFsequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?@21
/sequential_1/dense_12/kernel/Regularizer/Square?
.sequential_1/dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       20
.sequential_1/dense_12/kernel/Regularizer/Const?
,sequential_1/dense_12/kernel/Regularizer/SumSum3sequential_1/dense_12/kernel/Regularizer/Square:y:07sequential_1/dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2.
,sequential_1/dense_12/kernel/Regularizer/Sum?
.sequential_1/dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>20
.sequential_1/dense_12/kernel/Regularizer/mul/x?
,sequential_1/dense_12/kernel/Regularizer/mulMul7sequential_1/dense_12/kernel/Regularizer/mul/x:output:05sequential_1/dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,sequential_1/dense_12/kernel/Regularizer/mul?
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall?^sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp?^sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2?
>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp2?
>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp:Y U
'
_output_shapes
:?????????
*
_user_specified_namecategory_embed_1:YU
'
_output_shapes
:?????????
*
_user_specified_namecategory_embed_2:YU
'
_output_shapes
:?????????
*
_user_specified_namecategory_embed_3:YU
'
_output_shapes
:?????????
*
_user_specified_namecategory_embed_4:YU
'
_output_shapes
:?????????
*
_user_specified_namecategory_embed_5:UQ
'
_output_shapes
:?????????
&
_user_specified_namecity_embed_1:UQ
'
_output_shapes
:?????????
&
_user_specified_namecity_embed_2:UQ
'
_output_shapes
:?????????
&
_user_specified_namecity_embed_3:UQ
'
_output_shapes
:?????????
&
_user_specified_namecity_embed_4:U	Q
'
_output_shapes
:?????????
&
_user_specified_namecity_embed_5:N
J
'
_output_shapes
:?????????

_user_specified_namecolon:OK
'
_output_shapes
:?????????
 
_user_specified_namecommas:MI
'
_output_shapes
:?????????

_user_specified_namedash:OK
'
_output_shapes
:?????????
 
_user_specified_nameexclam:NJ
'
_output_shapes
:?????????

_user_specified_namemoney:NJ
'
_output_shapes
:?????????

_user_specified_namemonth:TP
'
_output_shapes
:?????????
%
_user_specified_nameparenthesis:VR
'
_output_shapes
:?????????
'
_user_specified_namestate_embed_1:VR
'
_output_shapes
:?????????
'
_user_specified_namestate_embed_2:VR
'
_output_shapes
:?????????
'
_user_specified_namestate_embed_3:VR
'
_output_shapes
:?????????
'
_user_specified_namestate_embed_4:VR
'
_output_shapes
:?????????
'
_user_specified_namestate_embed_5:PL
'
_output_shapes
:?????????
!
_user_specified_name	weekday
?

?
D__inference_dense_14_layer_call_and_return_conditional_losses_563884

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
L__inference_dense_features_1_layer_call_and_return_conditional_losses_563652
features_category_embed_1
features_category_embed_2
features_category_embed_3
features_category_embed_4
features_category_embed_5
features_city_embed_1
features_city_embed_2
features_city_embed_3
features_city_embed_4
features_city_embed_5
features_colon
features_commas
features_dash
features_exclam
features_money
features_month
features_parenthesis
features_state_embed_1
features_state_embed_2
features_state_embed_3
features_state_embed_4
features_state_embed_5
features_weekday
identityy
category_embed_1/ShapeShapefeatures_category_embed_1*
T0*
_output_shapes
:2
category_embed_1/Shape?
$category_embed_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$category_embed_1/strided_slice/stack?
&category_embed_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_1/strided_slice/stack_1?
&category_embed_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_1/strided_slice/stack_2?
category_embed_1/strided_sliceStridedSlicecategory_embed_1/Shape:output:0-category_embed_1/strided_slice/stack:output:0/category_embed_1/strided_slice/stack_1:output:0/category_embed_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
category_embed_1/strided_slice?
 category_embed_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 category_embed_1/Reshape/shape/1?
category_embed_1/Reshape/shapePack'category_embed_1/strided_slice:output:0)category_embed_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2 
category_embed_1/Reshape/shape?
category_embed_1/ReshapeReshapefeatures_category_embed_1'category_embed_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
category_embed_1/Reshapey
category_embed_2/ShapeShapefeatures_category_embed_2*
T0*
_output_shapes
:2
category_embed_2/Shape?
$category_embed_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$category_embed_2/strided_slice/stack?
&category_embed_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_2/strided_slice/stack_1?
&category_embed_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_2/strided_slice/stack_2?
category_embed_2/strided_sliceStridedSlicecategory_embed_2/Shape:output:0-category_embed_2/strided_slice/stack:output:0/category_embed_2/strided_slice/stack_1:output:0/category_embed_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
category_embed_2/strided_slice?
 category_embed_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 category_embed_2/Reshape/shape/1?
category_embed_2/Reshape/shapePack'category_embed_2/strided_slice:output:0)category_embed_2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2 
category_embed_2/Reshape/shape?
category_embed_2/ReshapeReshapefeatures_category_embed_2'category_embed_2/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
category_embed_2/Reshapey
category_embed_3/ShapeShapefeatures_category_embed_3*
T0*
_output_shapes
:2
category_embed_3/Shape?
$category_embed_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$category_embed_3/strided_slice/stack?
&category_embed_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_3/strided_slice/stack_1?
&category_embed_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_3/strided_slice/stack_2?
category_embed_3/strided_sliceStridedSlicecategory_embed_3/Shape:output:0-category_embed_3/strided_slice/stack:output:0/category_embed_3/strided_slice/stack_1:output:0/category_embed_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
category_embed_3/strided_slice?
 category_embed_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 category_embed_3/Reshape/shape/1?
category_embed_3/Reshape/shapePack'category_embed_3/strided_slice:output:0)category_embed_3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2 
category_embed_3/Reshape/shape?
category_embed_3/ReshapeReshapefeatures_category_embed_3'category_embed_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
category_embed_3/Reshapey
category_embed_4/ShapeShapefeatures_category_embed_4*
T0*
_output_shapes
:2
category_embed_4/Shape?
$category_embed_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$category_embed_4/strided_slice/stack?
&category_embed_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_4/strided_slice/stack_1?
&category_embed_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_4/strided_slice/stack_2?
category_embed_4/strided_sliceStridedSlicecategory_embed_4/Shape:output:0-category_embed_4/strided_slice/stack:output:0/category_embed_4/strided_slice/stack_1:output:0/category_embed_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
category_embed_4/strided_slice?
 category_embed_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 category_embed_4/Reshape/shape/1?
category_embed_4/Reshape/shapePack'category_embed_4/strided_slice:output:0)category_embed_4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2 
category_embed_4/Reshape/shape?
category_embed_4/ReshapeReshapefeatures_category_embed_4'category_embed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
category_embed_4/Reshapey
category_embed_5/ShapeShapefeatures_category_embed_5*
T0*
_output_shapes
:2
category_embed_5/Shape?
$category_embed_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$category_embed_5/strided_slice/stack?
&category_embed_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_5/strided_slice/stack_1?
&category_embed_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_5/strided_slice/stack_2?
category_embed_5/strided_sliceStridedSlicecategory_embed_5/Shape:output:0-category_embed_5/strided_slice/stack:output:0/category_embed_5/strided_slice/stack_1:output:0/category_embed_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
category_embed_5/strided_slice?
 category_embed_5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 category_embed_5/Reshape/shape/1?
category_embed_5/Reshape/shapePack'category_embed_5/strided_slice:output:0)category_embed_5/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2 
category_embed_5/Reshape/shape?
category_embed_5/ReshapeReshapefeatures_category_embed_5'category_embed_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
category_embed_5/Reshapem
city_embed_1/ShapeShapefeatures_city_embed_1*
T0*
_output_shapes
:2
city_embed_1/Shape?
 city_embed_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 city_embed_1/strided_slice/stack?
"city_embed_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_1/strided_slice/stack_1?
"city_embed_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_1/strided_slice/stack_2?
city_embed_1/strided_sliceStridedSlicecity_embed_1/Shape:output:0)city_embed_1/strided_slice/stack:output:0+city_embed_1/strided_slice/stack_1:output:0+city_embed_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
city_embed_1/strided_slice~
city_embed_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
city_embed_1/Reshape/shape/1?
city_embed_1/Reshape/shapePack#city_embed_1/strided_slice:output:0%city_embed_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
city_embed_1/Reshape/shape?
city_embed_1/ReshapeReshapefeatures_city_embed_1#city_embed_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
city_embed_1/Reshapem
city_embed_2/ShapeShapefeatures_city_embed_2*
T0*
_output_shapes
:2
city_embed_2/Shape?
 city_embed_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 city_embed_2/strided_slice/stack?
"city_embed_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_2/strided_slice/stack_1?
"city_embed_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_2/strided_slice/stack_2?
city_embed_2/strided_sliceStridedSlicecity_embed_2/Shape:output:0)city_embed_2/strided_slice/stack:output:0+city_embed_2/strided_slice/stack_1:output:0+city_embed_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
city_embed_2/strided_slice~
city_embed_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
city_embed_2/Reshape/shape/1?
city_embed_2/Reshape/shapePack#city_embed_2/strided_slice:output:0%city_embed_2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
city_embed_2/Reshape/shape?
city_embed_2/ReshapeReshapefeatures_city_embed_2#city_embed_2/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
city_embed_2/Reshapem
city_embed_3/ShapeShapefeatures_city_embed_3*
T0*
_output_shapes
:2
city_embed_3/Shape?
 city_embed_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 city_embed_3/strided_slice/stack?
"city_embed_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_3/strided_slice/stack_1?
"city_embed_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_3/strided_slice/stack_2?
city_embed_3/strided_sliceStridedSlicecity_embed_3/Shape:output:0)city_embed_3/strided_slice/stack:output:0+city_embed_3/strided_slice/stack_1:output:0+city_embed_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
city_embed_3/strided_slice~
city_embed_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
city_embed_3/Reshape/shape/1?
city_embed_3/Reshape/shapePack#city_embed_3/strided_slice:output:0%city_embed_3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
city_embed_3/Reshape/shape?
city_embed_3/ReshapeReshapefeatures_city_embed_3#city_embed_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
city_embed_3/Reshapem
city_embed_4/ShapeShapefeatures_city_embed_4*
T0*
_output_shapes
:2
city_embed_4/Shape?
 city_embed_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 city_embed_4/strided_slice/stack?
"city_embed_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_4/strided_slice/stack_1?
"city_embed_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_4/strided_slice/stack_2?
city_embed_4/strided_sliceStridedSlicecity_embed_4/Shape:output:0)city_embed_4/strided_slice/stack:output:0+city_embed_4/strided_slice/stack_1:output:0+city_embed_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
city_embed_4/strided_slice~
city_embed_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
city_embed_4/Reshape/shape/1?
city_embed_4/Reshape/shapePack#city_embed_4/strided_slice:output:0%city_embed_4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
city_embed_4/Reshape/shape?
city_embed_4/ReshapeReshapefeatures_city_embed_4#city_embed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
city_embed_4/Reshapem
city_embed_5/ShapeShapefeatures_city_embed_5*
T0*
_output_shapes
:2
city_embed_5/Shape?
 city_embed_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 city_embed_5/strided_slice/stack?
"city_embed_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_5/strided_slice/stack_1?
"city_embed_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_5/strided_slice/stack_2?
city_embed_5/strided_sliceStridedSlicecity_embed_5/Shape:output:0)city_embed_5/strided_slice/stack:output:0+city_embed_5/strided_slice/stack_1:output:0+city_embed_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
city_embed_5/strided_slice~
city_embed_5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
city_embed_5/Reshape/shape/1?
city_embed_5/Reshape/shapePack#city_embed_5/strided_slice:output:0%city_embed_5/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
city_embed_5/Reshape/shape?
city_embed_5/ReshapeReshapefeatures_city_embed_5#city_embed_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
city_embed_5/ReshapeX
colon/ShapeShapefeatures_colon*
T0*
_output_shapes
:2
colon/Shape?
colon/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
colon/strided_slice/stack?
colon/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
colon/strided_slice/stack_1?
colon/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
colon/strided_slice/stack_2?
colon/strided_sliceStridedSlicecolon/Shape:output:0"colon/strided_slice/stack:output:0$colon/strided_slice/stack_1:output:0$colon/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
colon/strided_slicep
colon/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
colon/Reshape/shape/1?
colon/Reshape/shapePackcolon/strided_slice:output:0colon/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
colon/Reshape/shape?
colon/ReshapeReshapefeatures_coloncolon/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
colon/Reshape[
commas/ShapeShapefeatures_commas*
T0*
_output_shapes
:2
commas/Shape?
commas/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
commas/strided_slice/stack?
commas/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
commas/strided_slice/stack_1?
commas/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
commas/strided_slice/stack_2?
commas/strided_sliceStridedSlicecommas/Shape:output:0#commas/strided_slice/stack:output:0%commas/strided_slice/stack_1:output:0%commas/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
commas/strided_slicer
commas/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
commas/Reshape/shape/1?
commas/Reshape/shapePackcommas/strided_slice:output:0commas/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
commas/Reshape/shape?
commas/ReshapeReshapefeatures_commascommas/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
commas/ReshapeU

dash/ShapeShapefeatures_dash*
T0*
_output_shapes
:2

dash/Shape~
dash/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
dash/strided_slice/stack?
dash/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
dash/strided_slice/stack_1?
dash/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
dash/strided_slice/stack_2?
dash/strided_sliceStridedSlicedash/Shape:output:0!dash/strided_slice/stack:output:0#dash/strided_slice/stack_1:output:0#dash/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dash/strided_slicen
dash/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
dash/Reshape/shape/1?
dash/Reshape/shapePackdash/strided_slice:output:0dash/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
dash/Reshape/shape?
dash/ReshapeReshapefeatures_dashdash/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
dash/Reshape[
exclam/ShapeShapefeatures_exclam*
T0*
_output_shapes
:2
exclam/Shape?
exclam/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
exclam/strided_slice/stack?
exclam/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
exclam/strided_slice/stack_1?
exclam/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
exclam/strided_slice/stack_2?
exclam/strided_sliceStridedSliceexclam/Shape:output:0#exclam/strided_slice/stack:output:0%exclam/strided_slice/stack_1:output:0%exclam/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
exclam/strided_slicer
exclam/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
exclam/Reshape/shape/1?
exclam/Reshape/shapePackexclam/strided_slice:output:0exclam/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
exclam/Reshape/shape?
exclam/ReshapeReshapefeatures_exclamexclam/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
exclam/ReshapeX
money/ShapeShapefeatures_money*
T0*
_output_shapes
:2
money/Shape?
money/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
money/strided_slice/stack?
money/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
money/strided_slice/stack_1?
money/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
money/strided_slice/stack_2?
money/strided_sliceStridedSlicemoney/Shape:output:0"money/strided_slice/stack:output:0$money/strided_slice/stack_1:output:0$money/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
money/strided_slicep
money/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
money/Reshape/shape/1?
money/Reshape/shapePackmoney/strided_slice:output:0money/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
money/Reshape/shape?
money/ReshapeReshapefeatures_moneymoney/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
money/ReshapeX
month/ShapeShapefeatures_month*
T0*
_output_shapes
:2
month/Shape?
month/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
month/strided_slice/stack?
month/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
month/strided_slice/stack_1?
month/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
month/strided_slice/stack_2?
month/strided_sliceStridedSlicemonth/Shape:output:0"month/strided_slice/stack:output:0$month/strided_slice/stack_1:output:0$month/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
month/strided_slicep
month/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
month/Reshape/shape/1?
month/Reshape/shapePackmonth/strided_slice:output:0month/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
month/Reshape/shape?
month/ReshapeReshapefeatures_monthmonth/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
month/Reshapej
parenthesis/ShapeShapefeatures_parenthesis*
T0*
_output_shapes
:2
parenthesis/Shape?
parenthesis/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
parenthesis/strided_slice/stack?
!parenthesis/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!parenthesis/strided_slice/stack_1?
!parenthesis/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!parenthesis/strided_slice/stack_2?
parenthesis/strided_sliceStridedSliceparenthesis/Shape:output:0(parenthesis/strided_slice/stack:output:0*parenthesis/strided_slice/stack_1:output:0*parenthesis/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
parenthesis/strided_slice|
parenthesis/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
parenthesis/Reshape/shape/1?
parenthesis/Reshape/shapePack"parenthesis/strided_slice:output:0$parenthesis/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
parenthesis/Reshape/shape?
parenthesis/ReshapeReshapefeatures_parenthesis"parenthesis/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
parenthesis/Reshapep
state_embed_1/ShapeShapefeatures_state_embed_1*
T0*
_output_shapes
:2
state_embed_1/Shape?
!state_embed_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!state_embed_1/strided_slice/stack?
#state_embed_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_1/strided_slice/stack_1?
#state_embed_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_1/strided_slice/stack_2?
state_embed_1/strided_sliceStridedSlicestate_embed_1/Shape:output:0*state_embed_1/strided_slice/stack:output:0,state_embed_1/strided_slice/stack_1:output:0,state_embed_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
state_embed_1/strided_slice?
state_embed_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
state_embed_1/Reshape/shape/1?
state_embed_1/Reshape/shapePack$state_embed_1/strided_slice:output:0&state_embed_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
state_embed_1/Reshape/shape?
state_embed_1/ReshapeReshapefeatures_state_embed_1$state_embed_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
state_embed_1/Reshapep
state_embed_2/ShapeShapefeatures_state_embed_2*
T0*
_output_shapes
:2
state_embed_2/Shape?
!state_embed_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!state_embed_2/strided_slice/stack?
#state_embed_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_2/strided_slice/stack_1?
#state_embed_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_2/strided_slice/stack_2?
state_embed_2/strided_sliceStridedSlicestate_embed_2/Shape:output:0*state_embed_2/strided_slice/stack:output:0,state_embed_2/strided_slice/stack_1:output:0,state_embed_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
state_embed_2/strided_slice?
state_embed_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
state_embed_2/Reshape/shape/1?
state_embed_2/Reshape/shapePack$state_embed_2/strided_slice:output:0&state_embed_2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
state_embed_2/Reshape/shape?
state_embed_2/ReshapeReshapefeatures_state_embed_2$state_embed_2/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
state_embed_2/Reshapep
state_embed_3/ShapeShapefeatures_state_embed_3*
T0*
_output_shapes
:2
state_embed_3/Shape?
!state_embed_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!state_embed_3/strided_slice/stack?
#state_embed_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_3/strided_slice/stack_1?
#state_embed_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_3/strided_slice/stack_2?
state_embed_3/strided_sliceStridedSlicestate_embed_3/Shape:output:0*state_embed_3/strided_slice/stack:output:0,state_embed_3/strided_slice/stack_1:output:0,state_embed_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
state_embed_3/strided_slice?
state_embed_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
state_embed_3/Reshape/shape/1?
state_embed_3/Reshape/shapePack$state_embed_3/strided_slice:output:0&state_embed_3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
state_embed_3/Reshape/shape?
state_embed_3/ReshapeReshapefeatures_state_embed_3$state_embed_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
state_embed_3/Reshapep
state_embed_4/ShapeShapefeatures_state_embed_4*
T0*
_output_shapes
:2
state_embed_4/Shape?
!state_embed_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!state_embed_4/strided_slice/stack?
#state_embed_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_4/strided_slice/stack_1?
#state_embed_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_4/strided_slice/stack_2?
state_embed_4/strided_sliceStridedSlicestate_embed_4/Shape:output:0*state_embed_4/strided_slice/stack:output:0,state_embed_4/strided_slice/stack_1:output:0,state_embed_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
state_embed_4/strided_slice?
state_embed_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
state_embed_4/Reshape/shape/1?
state_embed_4/Reshape/shapePack$state_embed_4/strided_slice:output:0&state_embed_4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
state_embed_4/Reshape/shape?
state_embed_4/ReshapeReshapefeatures_state_embed_4$state_embed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
state_embed_4/Reshapep
state_embed_5/ShapeShapefeatures_state_embed_5*
T0*
_output_shapes
:2
state_embed_5/Shape?
!state_embed_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!state_embed_5/strided_slice/stack?
#state_embed_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_5/strided_slice/stack_1?
#state_embed_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_5/strided_slice/stack_2?
state_embed_5/strided_sliceStridedSlicestate_embed_5/Shape:output:0*state_embed_5/strided_slice/stack:output:0,state_embed_5/strided_slice/stack_1:output:0,state_embed_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
state_embed_5/strided_slice?
state_embed_5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
state_embed_5/Reshape/shape/1?
state_embed_5/Reshape/shapePack$state_embed_5/strided_slice:output:0&state_embed_5/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
state_embed_5/Reshape/shape?
state_embed_5/ReshapeReshapefeatures_state_embed_5$state_embed_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
state_embed_5/Reshape^
weekday/ShapeShapefeatures_weekday*
T0*
_output_shapes
:2
weekday/Shape?
weekday/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
weekday/strided_slice/stack?
weekday/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
weekday/strided_slice/stack_1?
weekday/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
weekday/strided_slice/stack_2?
weekday/strided_sliceStridedSliceweekday/Shape:output:0$weekday/strided_slice/stack:output:0&weekday/strided_slice/stack_1:output:0&weekday/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
weekday/strided_slicet
weekday/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
weekday/Reshape/shape/1?
weekday/Reshape/shapePackweekday/strided_slice:output:0 weekday/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
weekday/Reshape/shape?
weekday/ReshapeReshapefeatures_weekdayweekday/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
weekday/Reshapee
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2!category_embed_1/Reshape:output:0!category_embed_2/Reshape:output:0!category_embed_3/Reshape:output:0!category_embed_4/Reshape:output:0!category_embed_5/Reshape:output:0city_embed_1/Reshape:output:0city_embed_2/Reshape:output:0city_embed_3/Reshape:output:0city_embed_4/Reshape:output:0city_embed_5/Reshape:output:0colon/Reshape:output:0commas/Reshape:output:0dash/Reshape:output:0exclam/Reshape:output:0money/Reshape:output:0month/Reshape:output:0parenthesis/Reshape:output:0state_embed_1/Reshape:output:0state_embed_2/Reshape:output:0state_embed_3/Reshape:output:0state_embed_4/Reshape:output:0state_embed_5/Reshape:output:0weekday/Reshape:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:b ^
'
_output_shapes
:?????????
3
_user_specified_namefeatures/category_embed_1:b^
'
_output_shapes
:?????????
3
_user_specified_namefeatures/category_embed_2:b^
'
_output_shapes
:?????????
3
_user_specified_namefeatures/category_embed_3:b^
'
_output_shapes
:?????????
3
_user_specified_namefeatures/category_embed_4:b^
'
_output_shapes
:?????????
3
_user_specified_namefeatures/category_embed_5:^Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/city_embed_1:^Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/city_embed_2:^Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/city_embed_3:^Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/city_embed_4:^	Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/city_embed_5:W
S
'
_output_shapes
:?????????
(
_user_specified_namefeatures/colon:XT
'
_output_shapes
:?????????
)
_user_specified_namefeatures/commas:VR
'
_output_shapes
:?????????
'
_user_specified_namefeatures/dash:XT
'
_output_shapes
:?????????
)
_user_specified_namefeatures/exclam:WS
'
_output_shapes
:?????????
(
_user_specified_namefeatures/money:WS
'
_output_shapes
:?????????
(
_user_specified_namefeatures/month:]Y
'
_output_shapes
:?????????
.
_user_specified_namefeatures/parenthesis:_[
'
_output_shapes
:?????????
0
_user_specified_namefeatures/state_embed_1:_[
'
_output_shapes
:?????????
0
_user_specified_namefeatures/state_embed_2:_[
'
_output_shapes
:?????????
0
_user_specified_namefeatures/state_embed_3:_[
'
_output_shapes
:?????????
0
_user_specified_namefeatures/state_embed_4:_[
'
_output_shapes
:?????????
0
_user_specified_namefeatures/state_embed_5:YU
'
_output_shapes
:?????????
*
_user_specified_namefeatures/weekday
??
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_563386
inputs_category_embed_1
inputs_category_embed_2
inputs_category_embed_3
inputs_category_embed_4
inputs_category_embed_5
inputs_city_embed_1
inputs_city_embed_2
inputs_city_embed_3
inputs_city_embed_4
inputs_city_embed_5
inputs_colon
inputs_commas
inputs_dash
inputs_exclam
inputs_money
inputs_month
inputs_parenthesis
inputs_state_embed_1
inputs_state_embed_2
inputs_state_embed_3
inputs_state_embed_4
inputs_state_embed_5
inputs_weekday9
'dense_14_matmul_readvariableop_resource: 6
(dense_14_biasadd_readvariableop_resource: L
>batch_normalization_11_assignmovingavg_readvariableop_resource: N
@batch_normalization_11_assignmovingavg_1_readvariableop_resource: A
3batch_normalization_11_cast_readvariableop_resource: C
5batch_normalization_11_cast_1_readvariableop_resource: :
'dense_13_matmul_readvariableop_resource:	 ?7
(dense_13_biasadd_readvariableop_resource:	?M
>batch_normalization_10_assignmovingavg_readvariableop_resource:	?O
@batch_normalization_10_assignmovingavg_1_readvariableop_resource:	?B
3batch_normalization_10_cast_readvariableop_resource:	?D
5batch_normalization_10_cast_1_readvariableop_resource:	?:
'dense_12_matmul_readvariableop_resource:	?@6
(dense_12_biasadd_readvariableop_resource:@K
=batch_normalization_9_assignmovingavg_readvariableop_resource:@M
?batch_normalization_9_assignmovingavg_1_readvariableop_resource:@@
2batch_normalization_9_cast_readvariableop_resource:@B
4batch_normalization_9_cast_1_readvariableop_resource:@:
'dense_11_matmul_readvariableop_resource:	@?7
(dense_11_biasadd_readvariableop_resource:	?L
=batch_normalization_8_assignmovingavg_readvariableop_resource:	?N
?batch_normalization_8_assignmovingavg_1_readvariableop_resource:	?A
2batch_normalization_8_cast_readvariableop_resource:	?C
4batch_normalization_8_cast_1_readvariableop_resource:	?:
'dense_10_matmul_readvariableop_resource:	?	6
(dense_10_biasadd_readvariableop_resource:	
identity??&batch_normalization_10/AssignMovingAvg?5batch_normalization_10/AssignMovingAvg/ReadVariableOp?(batch_normalization_10/AssignMovingAvg_1?7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp?*batch_normalization_10/Cast/ReadVariableOp?,batch_normalization_10/Cast_1/ReadVariableOp?&batch_normalization_11/AssignMovingAvg?5batch_normalization_11/AssignMovingAvg/ReadVariableOp?(batch_normalization_11/AssignMovingAvg_1?7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp?*batch_normalization_11/Cast/ReadVariableOp?,batch_normalization_11/Cast_1/ReadVariableOp?%batch_normalization_8/AssignMovingAvg?4batch_normalization_8/AssignMovingAvg/ReadVariableOp?'batch_normalization_8/AssignMovingAvg_1?6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp?)batch_normalization_8/Cast/ReadVariableOp?+batch_normalization_8/Cast_1/ReadVariableOp?%batch_normalization_9/AssignMovingAvg?4batch_normalization_9/AssignMovingAvg/ReadVariableOp?'batch_normalization_9/AssignMovingAvg_1?6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp?)batch_normalization_9/Cast/ReadVariableOp?+batch_normalization_9/Cast_1/ReadVariableOp?dense_10/BiasAdd/ReadVariableOp?dense_10/MatMul/ReadVariableOp?dense_11/BiasAdd/ReadVariableOp?dense_11/MatMul/ReadVariableOp?dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?dense_13/BiasAdd/ReadVariableOp?dense_13/MatMul/ReadVariableOp?dense_14/BiasAdd/ReadVariableOp?dense_14/MatMul/ReadVariableOp?>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp?>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp?
'dense_features_1/category_embed_1/ShapeShapeinputs_category_embed_1*
T0*
_output_shapes
:2)
'dense_features_1/category_embed_1/Shape?
5dense_features_1/category_embed_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5dense_features_1/category_embed_1/strided_slice/stack?
7dense_features_1/category_embed_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7dense_features_1/category_embed_1/strided_slice/stack_1?
7dense_features_1/category_embed_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7dense_features_1/category_embed_1/strided_slice/stack_2?
/dense_features_1/category_embed_1/strided_sliceStridedSlice0dense_features_1/category_embed_1/Shape:output:0>dense_features_1/category_embed_1/strided_slice/stack:output:0@dense_features_1/category_embed_1/strided_slice/stack_1:output:0@dense_features_1/category_embed_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/dense_features_1/category_embed_1/strided_slice?
1dense_features_1/category_embed_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :23
1dense_features_1/category_embed_1/Reshape/shape/1?
/dense_features_1/category_embed_1/Reshape/shapePack8dense_features_1/category_embed_1/strided_slice:output:0:dense_features_1/category_embed_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:21
/dense_features_1/category_embed_1/Reshape/shape?
)dense_features_1/category_embed_1/ReshapeReshapeinputs_category_embed_18dense_features_1/category_embed_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2+
)dense_features_1/category_embed_1/Reshape?
'dense_features_1/category_embed_2/ShapeShapeinputs_category_embed_2*
T0*
_output_shapes
:2)
'dense_features_1/category_embed_2/Shape?
5dense_features_1/category_embed_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5dense_features_1/category_embed_2/strided_slice/stack?
7dense_features_1/category_embed_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7dense_features_1/category_embed_2/strided_slice/stack_1?
7dense_features_1/category_embed_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7dense_features_1/category_embed_2/strided_slice/stack_2?
/dense_features_1/category_embed_2/strided_sliceStridedSlice0dense_features_1/category_embed_2/Shape:output:0>dense_features_1/category_embed_2/strided_slice/stack:output:0@dense_features_1/category_embed_2/strided_slice/stack_1:output:0@dense_features_1/category_embed_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/dense_features_1/category_embed_2/strided_slice?
1dense_features_1/category_embed_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :23
1dense_features_1/category_embed_2/Reshape/shape/1?
/dense_features_1/category_embed_2/Reshape/shapePack8dense_features_1/category_embed_2/strided_slice:output:0:dense_features_1/category_embed_2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:21
/dense_features_1/category_embed_2/Reshape/shape?
)dense_features_1/category_embed_2/ReshapeReshapeinputs_category_embed_28dense_features_1/category_embed_2/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2+
)dense_features_1/category_embed_2/Reshape?
'dense_features_1/category_embed_3/ShapeShapeinputs_category_embed_3*
T0*
_output_shapes
:2)
'dense_features_1/category_embed_3/Shape?
5dense_features_1/category_embed_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5dense_features_1/category_embed_3/strided_slice/stack?
7dense_features_1/category_embed_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7dense_features_1/category_embed_3/strided_slice/stack_1?
7dense_features_1/category_embed_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7dense_features_1/category_embed_3/strided_slice/stack_2?
/dense_features_1/category_embed_3/strided_sliceStridedSlice0dense_features_1/category_embed_3/Shape:output:0>dense_features_1/category_embed_3/strided_slice/stack:output:0@dense_features_1/category_embed_3/strided_slice/stack_1:output:0@dense_features_1/category_embed_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/dense_features_1/category_embed_3/strided_slice?
1dense_features_1/category_embed_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :23
1dense_features_1/category_embed_3/Reshape/shape/1?
/dense_features_1/category_embed_3/Reshape/shapePack8dense_features_1/category_embed_3/strided_slice:output:0:dense_features_1/category_embed_3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:21
/dense_features_1/category_embed_3/Reshape/shape?
)dense_features_1/category_embed_3/ReshapeReshapeinputs_category_embed_38dense_features_1/category_embed_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2+
)dense_features_1/category_embed_3/Reshape?
'dense_features_1/category_embed_4/ShapeShapeinputs_category_embed_4*
T0*
_output_shapes
:2)
'dense_features_1/category_embed_4/Shape?
5dense_features_1/category_embed_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5dense_features_1/category_embed_4/strided_slice/stack?
7dense_features_1/category_embed_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7dense_features_1/category_embed_4/strided_slice/stack_1?
7dense_features_1/category_embed_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7dense_features_1/category_embed_4/strided_slice/stack_2?
/dense_features_1/category_embed_4/strided_sliceStridedSlice0dense_features_1/category_embed_4/Shape:output:0>dense_features_1/category_embed_4/strided_slice/stack:output:0@dense_features_1/category_embed_4/strided_slice/stack_1:output:0@dense_features_1/category_embed_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/dense_features_1/category_embed_4/strided_slice?
1dense_features_1/category_embed_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :23
1dense_features_1/category_embed_4/Reshape/shape/1?
/dense_features_1/category_embed_4/Reshape/shapePack8dense_features_1/category_embed_4/strided_slice:output:0:dense_features_1/category_embed_4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:21
/dense_features_1/category_embed_4/Reshape/shape?
)dense_features_1/category_embed_4/ReshapeReshapeinputs_category_embed_48dense_features_1/category_embed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2+
)dense_features_1/category_embed_4/Reshape?
'dense_features_1/category_embed_5/ShapeShapeinputs_category_embed_5*
T0*
_output_shapes
:2)
'dense_features_1/category_embed_5/Shape?
5dense_features_1/category_embed_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5dense_features_1/category_embed_5/strided_slice/stack?
7dense_features_1/category_embed_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7dense_features_1/category_embed_5/strided_slice/stack_1?
7dense_features_1/category_embed_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7dense_features_1/category_embed_5/strided_slice/stack_2?
/dense_features_1/category_embed_5/strided_sliceStridedSlice0dense_features_1/category_embed_5/Shape:output:0>dense_features_1/category_embed_5/strided_slice/stack:output:0@dense_features_1/category_embed_5/strided_slice/stack_1:output:0@dense_features_1/category_embed_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/dense_features_1/category_embed_5/strided_slice?
1dense_features_1/category_embed_5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :23
1dense_features_1/category_embed_5/Reshape/shape/1?
/dense_features_1/category_embed_5/Reshape/shapePack8dense_features_1/category_embed_5/strided_slice:output:0:dense_features_1/category_embed_5/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:21
/dense_features_1/category_embed_5/Reshape/shape?
)dense_features_1/category_embed_5/ReshapeReshapeinputs_category_embed_58dense_features_1/category_embed_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2+
)dense_features_1/category_embed_5/Reshape?
#dense_features_1/city_embed_1/ShapeShapeinputs_city_embed_1*
T0*
_output_shapes
:2%
#dense_features_1/city_embed_1/Shape?
1dense_features_1/city_embed_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_features_1/city_embed_1/strided_slice/stack?
3dense_features_1/city_embed_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_features_1/city_embed_1/strided_slice/stack_1?
3dense_features_1/city_embed_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_features_1/city_embed_1/strided_slice/stack_2?
+dense_features_1/city_embed_1/strided_sliceStridedSlice,dense_features_1/city_embed_1/Shape:output:0:dense_features_1/city_embed_1/strided_slice/stack:output:0<dense_features_1/city_embed_1/strided_slice/stack_1:output:0<dense_features_1/city_embed_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_features_1/city_embed_1/strided_slice?
-dense_features_1/city_embed_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2/
-dense_features_1/city_embed_1/Reshape/shape/1?
+dense_features_1/city_embed_1/Reshape/shapePack4dense_features_1/city_embed_1/strided_slice:output:06dense_features_1/city_embed_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2-
+dense_features_1/city_embed_1/Reshape/shape?
%dense_features_1/city_embed_1/ReshapeReshapeinputs_city_embed_14dense_features_1/city_embed_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2'
%dense_features_1/city_embed_1/Reshape?
#dense_features_1/city_embed_2/ShapeShapeinputs_city_embed_2*
T0*
_output_shapes
:2%
#dense_features_1/city_embed_2/Shape?
1dense_features_1/city_embed_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_features_1/city_embed_2/strided_slice/stack?
3dense_features_1/city_embed_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_features_1/city_embed_2/strided_slice/stack_1?
3dense_features_1/city_embed_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_features_1/city_embed_2/strided_slice/stack_2?
+dense_features_1/city_embed_2/strided_sliceStridedSlice,dense_features_1/city_embed_2/Shape:output:0:dense_features_1/city_embed_2/strided_slice/stack:output:0<dense_features_1/city_embed_2/strided_slice/stack_1:output:0<dense_features_1/city_embed_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_features_1/city_embed_2/strided_slice?
-dense_features_1/city_embed_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2/
-dense_features_1/city_embed_2/Reshape/shape/1?
+dense_features_1/city_embed_2/Reshape/shapePack4dense_features_1/city_embed_2/strided_slice:output:06dense_features_1/city_embed_2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2-
+dense_features_1/city_embed_2/Reshape/shape?
%dense_features_1/city_embed_2/ReshapeReshapeinputs_city_embed_24dense_features_1/city_embed_2/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2'
%dense_features_1/city_embed_2/Reshape?
#dense_features_1/city_embed_3/ShapeShapeinputs_city_embed_3*
T0*
_output_shapes
:2%
#dense_features_1/city_embed_3/Shape?
1dense_features_1/city_embed_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_features_1/city_embed_3/strided_slice/stack?
3dense_features_1/city_embed_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_features_1/city_embed_3/strided_slice/stack_1?
3dense_features_1/city_embed_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_features_1/city_embed_3/strided_slice/stack_2?
+dense_features_1/city_embed_3/strided_sliceStridedSlice,dense_features_1/city_embed_3/Shape:output:0:dense_features_1/city_embed_3/strided_slice/stack:output:0<dense_features_1/city_embed_3/strided_slice/stack_1:output:0<dense_features_1/city_embed_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_features_1/city_embed_3/strided_slice?
-dense_features_1/city_embed_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2/
-dense_features_1/city_embed_3/Reshape/shape/1?
+dense_features_1/city_embed_3/Reshape/shapePack4dense_features_1/city_embed_3/strided_slice:output:06dense_features_1/city_embed_3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2-
+dense_features_1/city_embed_3/Reshape/shape?
%dense_features_1/city_embed_3/ReshapeReshapeinputs_city_embed_34dense_features_1/city_embed_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2'
%dense_features_1/city_embed_3/Reshape?
#dense_features_1/city_embed_4/ShapeShapeinputs_city_embed_4*
T0*
_output_shapes
:2%
#dense_features_1/city_embed_4/Shape?
1dense_features_1/city_embed_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_features_1/city_embed_4/strided_slice/stack?
3dense_features_1/city_embed_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_features_1/city_embed_4/strided_slice/stack_1?
3dense_features_1/city_embed_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_features_1/city_embed_4/strided_slice/stack_2?
+dense_features_1/city_embed_4/strided_sliceStridedSlice,dense_features_1/city_embed_4/Shape:output:0:dense_features_1/city_embed_4/strided_slice/stack:output:0<dense_features_1/city_embed_4/strided_slice/stack_1:output:0<dense_features_1/city_embed_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_features_1/city_embed_4/strided_slice?
-dense_features_1/city_embed_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2/
-dense_features_1/city_embed_4/Reshape/shape/1?
+dense_features_1/city_embed_4/Reshape/shapePack4dense_features_1/city_embed_4/strided_slice:output:06dense_features_1/city_embed_4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2-
+dense_features_1/city_embed_4/Reshape/shape?
%dense_features_1/city_embed_4/ReshapeReshapeinputs_city_embed_44dense_features_1/city_embed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2'
%dense_features_1/city_embed_4/Reshape?
#dense_features_1/city_embed_5/ShapeShapeinputs_city_embed_5*
T0*
_output_shapes
:2%
#dense_features_1/city_embed_5/Shape?
1dense_features_1/city_embed_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_features_1/city_embed_5/strided_slice/stack?
3dense_features_1/city_embed_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_features_1/city_embed_5/strided_slice/stack_1?
3dense_features_1/city_embed_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_features_1/city_embed_5/strided_slice/stack_2?
+dense_features_1/city_embed_5/strided_sliceStridedSlice,dense_features_1/city_embed_5/Shape:output:0:dense_features_1/city_embed_5/strided_slice/stack:output:0<dense_features_1/city_embed_5/strided_slice/stack_1:output:0<dense_features_1/city_embed_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_features_1/city_embed_5/strided_slice?
-dense_features_1/city_embed_5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2/
-dense_features_1/city_embed_5/Reshape/shape/1?
+dense_features_1/city_embed_5/Reshape/shapePack4dense_features_1/city_embed_5/strided_slice:output:06dense_features_1/city_embed_5/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2-
+dense_features_1/city_embed_5/Reshape/shape?
%dense_features_1/city_embed_5/ReshapeReshapeinputs_city_embed_54dense_features_1/city_embed_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2'
%dense_features_1/city_embed_5/Reshapex
dense_features_1/colon/ShapeShapeinputs_colon*
T0*
_output_shapes
:2
dense_features_1/colon/Shape?
*dense_features_1/colon/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*dense_features_1/colon/strided_slice/stack?
,dense_features_1/colon/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,dense_features_1/colon/strided_slice/stack_1?
,dense_features_1/colon/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,dense_features_1/colon/strided_slice/stack_2?
$dense_features_1/colon/strided_sliceStridedSlice%dense_features_1/colon/Shape:output:03dense_features_1/colon/strided_slice/stack:output:05dense_features_1/colon/strided_slice/stack_1:output:05dense_features_1/colon/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$dense_features_1/colon/strided_slice?
&dense_features_1/colon/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&dense_features_1/colon/Reshape/shape/1?
$dense_features_1/colon/Reshape/shapePack-dense_features_1/colon/strided_slice:output:0/dense_features_1/colon/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2&
$dense_features_1/colon/Reshape/shape?
dense_features_1/colon/ReshapeReshapeinputs_colon-dense_features_1/colon/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2 
dense_features_1/colon/Reshape{
dense_features_1/commas/ShapeShapeinputs_commas*
T0*
_output_shapes
:2
dense_features_1/commas/Shape?
+dense_features_1/commas/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+dense_features_1/commas/strided_slice/stack?
-dense_features_1/commas/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-dense_features_1/commas/strided_slice/stack_1?
-dense_features_1/commas/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-dense_features_1/commas/strided_slice/stack_2?
%dense_features_1/commas/strided_sliceStridedSlice&dense_features_1/commas/Shape:output:04dense_features_1/commas/strided_slice/stack:output:06dense_features_1/commas/strided_slice/stack_1:output:06dense_features_1/commas/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%dense_features_1/commas/strided_slice?
'dense_features_1/commas/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'dense_features_1/commas/Reshape/shape/1?
%dense_features_1/commas/Reshape/shapePack.dense_features_1/commas/strided_slice:output:00dense_features_1/commas/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2'
%dense_features_1/commas/Reshape/shape?
dense_features_1/commas/ReshapeReshapeinputs_commas.dense_features_1/commas/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2!
dense_features_1/commas/Reshapeu
dense_features_1/dash/ShapeShapeinputs_dash*
T0*
_output_shapes
:2
dense_features_1/dash/Shape?
)dense_features_1/dash/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)dense_features_1/dash/strided_slice/stack?
+dense_features_1/dash/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+dense_features_1/dash/strided_slice/stack_1?
+dense_features_1/dash/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+dense_features_1/dash/strided_slice/stack_2?
#dense_features_1/dash/strided_sliceStridedSlice$dense_features_1/dash/Shape:output:02dense_features_1/dash/strided_slice/stack:output:04dense_features_1/dash/strided_slice/stack_1:output:04dense_features_1/dash/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#dense_features_1/dash/strided_slice?
%dense_features_1/dash/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%dense_features_1/dash/Reshape/shape/1?
#dense_features_1/dash/Reshape/shapePack,dense_features_1/dash/strided_slice:output:0.dense_features_1/dash/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2%
#dense_features_1/dash/Reshape/shape?
dense_features_1/dash/ReshapeReshapeinputs_dash,dense_features_1/dash/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
dense_features_1/dash/Reshape{
dense_features_1/exclam/ShapeShapeinputs_exclam*
T0*
_output_shapes
:2
dense_features_1/exclam/Shape?
+dense_features_1/exclam/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+dense_features_1/exclam/strided_slice/stack?
-dense_features_1/exclam/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-dense_features_1/exclam/strided_slice/stack_1?
-dense_features_1/exclam/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-dense_features_1/exclam/strided_slice/stack_2?
%dense_features_1/exclam/strided_sliceStridedSlice&dense_features_1/exclam/Shape:output:04dense_features_1/exclam/strided_slice/stack:output:06dense_features_1/exclam/strided_slice/stack_1:output:06dense_features_1/exclam/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%dense_features_1/exclam/strided_slice?
'dense_features_1/exclam/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'dense_features_1/exclam/Reshape/shape/1?
%dense_features_1/exclam/Reshape/shapePack.dense_features_1/exclam/strided_slice:output:00dense_features_1/exclam/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2'
%dense_features_1/exclam/Reshape/shape?
dense_features_1/exclam/ReshapeReshapeinputs_exclam.dense_features_1/exclam/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2!
dense_features_1/exclam/Reshapex
dense_features_1/money/ShapeShapeinputs_money*
T0*
_output_shapes
:2
dense_features_1/money/Shape?
*dense_features_1/money/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*dense_features_1/money/strided_slice/stack?
,dense_features_1/money/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,dense_features_1/money/strided_slice/stack_1?
,dense_features_1/money/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,dense_features_1/money/strided_slice/stack_2?
$dense_features_1/money/strided_sliceStridedSlice%dense_features_1/money/Shape:output:03dense_features_1/money/strided_slice/stack:output:05dense_features_1/money/strided_slice/stack_1:output:05dense_features_1/money/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$dense_features_1/money/strided_slice?
&dense_features_1/money/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&dense_features_1/money/Reshape/shape/1?
$dense_features_1/money/Reshape/shapePack-dense_features_1/money/strided_slice:output:0/dense_features_1/money/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2&
$dense_features_1/money/Reshape/shape?
dense_features_1/money/ReshapeReshapeinputs_money-dense_features_1/money/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2 
dense_features_1/money/Reshapex
dense_features_1/month/ShapeShapeinputs_month*
T0*
_output_shapes
:2
dense_features_1/month/Shape?
*dense_features_1/month/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*dense_features_1/month/strided_slice/stack?
,dense_features_1/month/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,dense_features_1/month/strided_slice/stack_1?
,dense_features_1/month/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,dense_features_1/month/strided_slice/stack_2?
$dense_features_1/month/strided_sliceStridedSlice%dense_features_1/month/Shape:output:03dense_features_1/month/strided_slice/stack:output:05dense_features_1/month/strided_slice/stack_1:output:05dense_features_1/month/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$dense_features_1/month/strided_slice?
&dense_features_1/month/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&dense_features_1/month/Reshape/shape/1?
$dense_features_1/month/Reshape/shapePack-dense_features_1/month/strided_slice:output:0/dense_features_1/month/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2&
$dense_features_1/month/Reshape/shape?
dense_features_1/month/ReshapeReshapeinputs_month-dense_features_1/month/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2 
dense_features_1/month/Reshape?
"dense_features_1/parenthesis/ShapeShapeinputs_parenthesis*
T0*
_output_shapes
:2$
"dense_features_1/parenthesis/Shape?
0dense_features_1/parenthesis/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_features_1/parenthesis/strided_slice/stack?
2dense_features_1/parenthesis/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_features_1/parenthesis/strided_slice/stack_1?
2dense_features_1/parenthesis/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_features_1/parenthesis/strided_slice/stack_2?
*dense_features_1/parenthesis/strided_sliceStridedSlice+dense_features_1/parenthesis/Shape:output:09dense_features_1/parenthesis/strided_slice/stack:output:0;dense_features_1/parenthesis/strided_slice/stack_1:output:0;dense_features_1/parenthesis/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_features_1/parenthesis/strided_slice?
,dense_features_1/parenthesis/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2.
,dense_features_1/parenthesis/Reshape/shape/1?
*dense_features_1/parenthesis/Reshape/shapePack3dense_features_1/parenthesis/strided_slice:output:05dense_features_1/parenthesis/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2,
*dense_features_1/parenthesis/Reshape/shape?
$dense_features_1/parenthesis/ReshapeReshapeinputs_parenthesis3dense_features_1/parenthesis/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2&
$dense_features_1/parenthesis/Reshape?
$dense_features_1/state_embed_1/ShapeShapeinputs_state_embed_1*
T0*
_output_shapes
:2&
$dense_features_1/state_embed_1/Shape?
2dense_features_1/state_embed_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2dense_features_1/state_embed_1/strided_slice/stack?
4dense_features_1/state_embed_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4dense_features_1/state_embed_1/strided_slice/stack_1?
4dense_features_1/state_embed_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4dense_features_1/state_embed_1/strided_slice/stack_2?
,dense_features_1/state_embed_1/strided_sliceStridedSlice-dense_features_1/state_embed_1/Shape:output:0;dense_features_1/state_embed_1/strided_slice/stack:output:0=dense_features_1/state_embed_1/strided_slice/stack_1:output:0=dense_features_1/state_embed_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,dense_features_1/state_embed_1/strided_slice?
.dense_features_1/state_embed_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :20
.dense_features_1/state_embed_1/Reshape/shape/1?
,dense_features_1/state_embed_1/Reshape/shapePack5dense_features_1/state_embed_1/strided_slice:output:07dense_features_1/state_embed_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2.
,dense_features_1/state_embed_1/Reshape/shape?
&dense_features_1/state_embed_1/ReshapeReshapeinputs_state_embed_15dense_features_1/state_embed_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2(
&dense_features_1/state_embed_1/Reshape?
$dense_features_1/state_embed_2/ShapeShapeinputs_state_embed_2*
T0*
_output_shapes
:2&
$dense_features_1/state_embed_2/Shape?
2dense_features_1/state_embed_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2dense_features_1/state_embed_2/strided_slice/stack?
4dense_features_1/state_embed_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4dense_features_1/state_embed_2/strided_slice/stack_1?
4dense_features_1/state_embed_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4dense_features_1/state_embed_2/strided_slice/stack_2?
,dense_features_1/state_embed_2/strided_sliceStridedSlice-dense_features_1/state_embed_2/Shape:output:0;dense_features_1/state_embed_2/strided_slice/stack:output:0=dense_features_1/state_embed_2/strided_slice/stack_1:output:0=dense_features_1/state_embed_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,dense_features_1/state_embed_2/strided_slice?
.dense_features_1/state_embed_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :20
.dense_features_1/state_embed_2/Reshape/shape/1?
,dense_features_1/state_embed_2/Reshape/shapePack5dense_features_1/state_embed_2/strided_slice:output:07dense_features_1/state_embed_2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2.
,dense_features_1/state_embed_2/Reshape/shape?
&dense_features_1/state_embed_2/ReshapeReshapeinputs_state_embed_25dense_features_1/state_embed_2/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2(
&dense_features_1/state_embed_2/Reshape?
$dense_features_1/state_embed_3/ShapeShapeinputs_state_embed_3*
T0*
_output_shapes
:2&
$dense_features_1/state_embed_3/Shape?
2dense_features_1/state_embed_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2dense_features_1/state_embed_3/strided_slice/stack?
4dense_features_1/state_embed_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4dense_features_1/state_embed_3/strided_slice/stack_1?
4dense_features_1/state_embed_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4dense_features_1/state_embed_3/strided_slice/stack_2?
,dense_features_1/state_embed_3/strided_sliceStridedSlice-dense_features_1/state_embed_3/Shape:output:0;dense_features_1/state_embed_3/strided_slice/stack:output:0=dense_features_1/state_embed_3/strided_slice/stack_1:output:0=dense_features_1/state_embed_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,dense_features_1/state_embed_3/strided_slice?
.dense_features_1/state_embed_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :20
.dense_features_1/state_embed_3/Reshape/shape/1?
,dense_features_1/state_embed_3/Reshape/shapePack5dense_features_1/state_embed_3/strided_slice:output:07dense_features_1/state_embed_3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2.
,dense_features_1/state_embed_3/Reshape/shape?
&dense_features_1/state_embed_3/ReshapeReshapeinputs_state_embed_35dense_features_1/state_embed_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2(
&dense_features_1/state_embed_3/Reshape?
$dense_features_1/state_embed_4/ShapeShapeinputs_state_embed_4*
T0*
_output_shapes
:2&
$dense_features_1/state_embed_4/Shape?
2dense_features_1/state_embed_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2dense_features_1/state_embed_4/strided_slice/stack?
4dense_features_1/state_embed_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4dense_features_1/state_embed_4/strided_slice/stack_1?
4dense_features_1/state_embed_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4dense_features_1/state_embed_4/strided_slice/stack_2?
,dense_features_1/state_embed_4/strided_sliceStridedSlice-dense_features_1/state_embed_4/Shape:output:0;dense_features_1/state_embed_4/strided_slice/stack:output:0=dense_features_1/state_embed_4/strided_slice/stack_1:output:0=dense_features_1/state_embed_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,dense_features_1/state_embed_4/strided_slice?
.dense_features_1/state_embed_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :20
.dense_features_1/state_embed_4/Reshape/shape/1?
,dense_features_1/state_embed_4/Reshape/shapePack5dense_features_1/state_embed_4/strided_slice:output:07dense_features_1/state_embed_4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2.
,dense_features_1/state_embed_4/Reshape/shape?
&dense_features_1/state_embed_4/ReshapeReshapeinputs_state_embed_45dense_features_1/state_embed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2(
&dense_features_1/state_embed_4/Reshape?
$dense_features_1/state_embed_5/ShapeShapeinputs_state_embed_5*
T0*
_output_shapes
:2&
$dense_features_1/state_embed_5/Shape?
2dense_features_1/state_embed_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2dense_features_1/state_embed_5/strided_slice/stack?
4dense_features_1/state_embed_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4dense_features_1/state_embed_5/strided_slice/stack_1?
4dense_features_1/state_embed_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4dense_features_1/state_embed_5/strided_slice/stack_2?
,dense_features_1/state_embed_5/strided_sliceStridedSlice-dense_features_1/state_embed_5/Shape:output:0;dense_features_1/state_embed_5/strided_slice/stack:output:0=dense_features_1/state_embed_5/strided_slice/stack_1:output:0=dense_features_1/state_embed_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,dense_features_1/state_embed_5/strided_slice?
.dense_features_1/state_embed_5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :20
.dense_features_1/state_embed_5/Reshape/shape/1?
,dense_features_1/state_embed_5/Reshape/shapePack5dense_features_1/state_embed_5/strided_slice:output:07dense_features_1/state_embed_5/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2.
,dense_features_1/state_embed_5/Reshape/shape?
&dense_features_1/state_embed_5/ReshapeReshapeinputs_state_embed_55dense_features_1/state_embed_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2(
&dense_features_1/state_embed_5/Reshape~
dense_features_1/weekday/ShapeShapeinputs_weekday*
T0*
_output_shapes
:2 
dense_features_1/weekday/Shape?
,dense_features_1/weekday/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,dense_features_1/weekday/strided_slice/stack?
.dense_features_1/weekday/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.dense_features_1/weekday/strided_slice/stack_1?
.dense_features_1/weekday/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.dense_features_1/weekday/strided_slice/stack_2?
&dense_features_1/weekday/strided_sliceStridedSlice'dense_features_1/weekday/Shape:output:05dense_features_1/weekday/strided_slice/stack:output:07dense_features_1/weekday/strided_slice/stack_1:output:07dense_features_1/weekday/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&dense_features_1/weekday/strided_slice?
(dense_features_1/weekday/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(dense_features_1/weekday/Reshape/shape/1?
&dense_features_1/weekday/Reshape/shapePack/dense_features_1/weekday/strided_slice:output:01dense_features_1/weekday/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2(
&dense_features_1/weekday/Reshape/shape?
 dense_features_1/weekday/ReshapeReshapeinputs_weekday/dense_features_1/weekday/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2"
 dense_features_1/weekday/Reshape?
dense_features_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
dense_features_1/concat/axis?	
dense_features_1/concatConcatV22dense_features_1/category_embed_1/Reshape:output:02dense_features_1/category_embed_2/Reshape:output:02dense_features_1/category_embed_3/Reshape:output:02dense_features_1/category_embed_4/Reshape:output:02dense_features_1/category_embed_5/Reshape:output:0.dense_features_1/city_embed_1/Reshape:output:0.dense_features_1/city_embed_2/Reshape:output:0.dense_features_1/city_embed_3/Reshape:output:0.dense_features_1/city_embed_4/Reshape:output:0.dense_features_1/city_embed_5/Reshape:output:0'dense_features_1/colon/Reshape:output:0(dense_features_1/commas/Reshape:output:0&dense_features_1/dash/Reshape:output:0(dense_features_1/exclam/Reshape:output:0'dense_features_1/money/Reshape:output:0'dense_features_1/month/Reshape:output:0-dense_features_1/parenthesis/Reshape:output:0/dense_features_1/state_embed_1/Reshape:output:0/dense_features_1/state_embed_2/Reshape:output:0/dense_features_1/state_embed_3/Reshape:output:0/dense_features_1/state_embed_4/Reshape:output:0/dense_features_1/state_embed_5/Reshape:output:0)dense_features_1/weekday/Reshape:output:0%dense_features_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
dense_features_1/concat?
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_14/MatMul/ReadVariableOp?
dense_14/MatMulMatMul dense_features_1/concat:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_14/MatMul?
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_14/BiasAdd/ReadVariableOp?
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_14/BiasAdds
dense_14/ReluReludense_14/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_14/Relu?
5batch_normalization_11/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_11/moments/mean/reduction_indices?
#batch_normalization_11/moments/meanMeandense_14/Relu:activations:0>batch_normalization_11/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(2%
#batch_normalization_11/moments/mean?
+batch_normalization_11/moments/StopGradientStopGradient,batch_normalization_11/moments/mean:output:0*
T0*
_output_shapes

: 2-
+batch_normalization_11/moments/StopGradient?
0batch_normalization_11/moments/SquaredDifferenceSquaredDifferencedense_14/Relu:activations:04batch_normalization_11/moments/StopGradient:output:0*
T0*'
_output_shapes
:????????? 22
0batch_normalization_11/moments/SquaredDifference?
9batch_normalization_11/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_11/moments/variance/reduction_indices?
'batch_normalization_11/moments/varianceMean4batch_normalization_11/moments/SquaredDifference:z:0Bbatch_normalization_11/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(2)
'batch_normalization_11/moments/variance?
&batch_normalization_11/moments/SqueezeSqueeze,batch_normalization_11/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2(
&batch_normalization_11/moments/Squeeze?
(batch_normalization_11/moments/Squeeze_1Squeeze0batch_normalization_11/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2*
(batch_normalization_11/moments/Squeeze_1?
,batch_normalization_11/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_11/AssignMovingAvg/decay?
5batch_normalization_11/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_11_assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_11/AssignMovingAvg/ReadVariableOp?
*batch_normalization_11/AssignMovingAvg/subSub=batch_normalization_11/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_11/moments/Squeeze:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_11/AssignMovingAvg/sub?
*batch_normalization_11/AssignMovingAvg/mulMul.batch_normalization_11/AssignMovingAvg/sub:z:05batch_normalization_11/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_11/AssignMovingAvg/mul?
&batch_normalization_11/AssignMovingAvgAssignSubVariableOp>batch_normalization_11_assignmovingavg_readvariableop_resource.batch_normalization_11/AssignMovingAvg/mul:z:06^batch_normalization_11/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_11/AssignMovingAvg?
.batch_normalization_11/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_11/AssignMovingAvg_1/decay?
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_11_assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype029
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_11/AssignMovingAvg_1/subSub?batch_normalization_11/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_11/moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2.
,batch_normalization_11/AssignMovingAvg_1/sub?
,batch_normalization_11/AssignMovingAvg_1/mulMul0batch_normalization_11/AssignMovingAvg_1/sub:z:07batch_normalization_11/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2.
,batch_normalization_11/AssignMovingAvg_1/mul?
(batch_normalization_11/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_11_assignmovingavg_1_readvariableop_resource0batch_normalization_11/AssignMovingAvg_1/mul:z:08^batch_normalization_11/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_11/AssignMovingAvg_1?
*batch_normalization_11/Cast/ReadVariableOpReadVariableOp3batch_normalization_11_cast_readvariableop_resource*
_output_shapes
: *
dtype02,
*batch_normalization_11/Cast/ReadVariableOp?
,batch_normalization_11/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_11_cast_1_readvariableop_resource*
_output_shapes
: *
dtype02.
,batch_normalization_11/Cast_1/ReadVariableOp?
&batch_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_11/batchnorm/add/y?
$batch_normalization_11/batchnorm/addAddV21batch_normalization_11/moments/Squeeze_1:output:0/batch_normalization_11/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2&
$batch_normalization_11/batchnorm/add?
&batch_normalization_11/batchnorm/RsqrtRsqrt(batch_normalization_11/batchnorm/add:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_11/batchnorm/Rsqrt?
$batch_normalization_11/batchnorm/mulMul*batch_normalization_11/batchnorm/Rsqrt:y:04batch_normalization_11/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
: 2&
$batch_normalization_11/batchnorm/mul?
&batch_normalization_11/batchnorm/mul_1Muldense_14/Relu:activations:0(batch_normalization_11/batchnorm/mul:z:0*
T0*'
_output_shapes
:????????? 2(
&batch_normalization_11/batchnorm/mul_1?
&batch_normalization_11/batchnorm/mul_2Mul/batch_normalization_11/moments/Squeeze:output:0(batch_normalization_11/batchnorm/mul:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_11/batchnorm/mul_2?
$batch_normalization_11/batchnorm/subSub2batch_normalization_11/Cast/ReadVariableOp:value:0*batch_normalization_11/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2&
$batch_normalization_11/batchnorm/sub?
&batch_normalization_11/batchnorm/add_1AddV2*batch_normalization_11/batchnorm/mul_1:z:0(batch_normalization_11/batchnorm/sub:z:0*
T0*'
_output_shapes
:????????? 2(
&batch_normalization_11/batchnorm/add_1?
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02 
dense_13/MatMul/ReadVariableOp?
dense_13/MatMulMatMul*batch_normalization_11/batchnorm/add_1:z:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_13/MatMul?
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_13/BiasAdd/ReadVariableOp?
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_13/BiasAddt
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_13/Relu?
5batch_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_10/moments/mean/reduction_indices?
#batch_normalization_10/moments/meanMeandense_13/Relu:activations:0>batch_normalization_10/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2%
#batch_normalization_10/moments/mean?
+batch_normalization_10/moments/StopGradientStopGradient,batch_normalization_10/moments/mean:output:0*
T0*
_output_shapes
:	?2-
+batch_normalization_10/moments/StopGradient?
0batch_normalization_10/moments/SquaredDifferenceSquaredDifferencedense_13/Relu:activations:04batch_normalization_10/moments/StopGradient:output:0*
T0*(
_output_shapes
:??????????22
0batch_normalization_10/moments/SquaredDifference?
9batch_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_10/moments/variance/reduction_indices?
'batch_normalization_10/moments/varianceMean4batch_normalization_10/moments/SquaredDifference:z:0Bbatch_normalization_10/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2)
'batch_normalization_10/moments/variance?
&batch_normalization_10/moments/SqueezeSqueeze,batch_normalization_10/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2(
&batch_normalization_10/moments/Squeeze?
(batch_normalization_10/moments/Squeeze_1Squeeze0batch_normalization_10/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2*
(batch_normalization_10/moments/Squeeze_1?
,batch_normalization_10/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_10/AssignMovingAvg/decay?
5batch_normalization_10/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_10_assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype027
5batch_normalization_10/AssignMovingAvg/ReadVariableOp?
*batch_normalization_10/AssignMovingAvg/subSub=batch_normalization_10/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_10/moments/Squeeze:output:0*
T0*
_output_shapes	
:?2,
*batch_normalization_10/AssignMovingAvg/sub?
*batch_normalization_10/AssignMovingAvg/mulMul.batch_normalization_10/AssignMovingAvg/sub:z:05batch_normalization_10/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?2,
*batch_normalization_10/AssignMovingAvg/mul?
&batch_normalization_10/AssignMovingAvgAssignSubVariableOp>batch_normalization_10_assignmovingavg_readvariableop_resource.batch_normalization_10/AssignMovingAvg/mul:z:06^batch_normalization_10/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_10/AssignMovingAvg?
.batch_normalization_10/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_10/AssignMovingAvg_1/decay?
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_10_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype029
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_10/AssignMovingAvg_1/subSub?batch_normalization_10/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_10/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?2.
,batch_normalization_10/AssignMovingAvg_1/sub?
,batch_normalization_10/AssignMovingAvg_1/mulMul0batch_normalization_10/AssignMovingAvg_1/sub:z:07batch_normalization_10/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?2.
,batch_normalization_10/AssignMovingAvg_1/mul?
(batch_normalization_10/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_10_assignmovingavg_1_readvariableop_resource0batch_normalization_10/AssignMovingAvg_1/mul:z:08^batch_normalization_10/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_10/AssignMovingAvg_1?
*batch_normalization_10/Cast/ReadVariableOpReadVariableOp3batch_normalization_10_cast_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*batch_normalization_10/Cast/ReadVariableOp?
,batch_normalization_10/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_10_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,batch_normalization_10/Cast_1/ReadVariableOp?
&batch_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_10/batchnorm/add/y?
$batch_normalization_10/batchnorm/addAddV21batch_normalization_10/moments/Squeeze_1:output:0/batch_normalization_10/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2&
$batch_normalization_10/batchnorm/add?
&batch_normalization_10/batchnorm/RsqrtRsqrt(batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_10/batchnorm/Rsqrt?
$batch_normalization_10/batchnorm/mulMul*batch_normalization_10/batchnorm/Rsqrt:y:04batch_normalization_10/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2&
$batch_normalization_10/batchnorm/mul?
&batch_normalization_10/batchnorm/mul_1Muldense_13/Relu:activations:0(batch_normalization_10/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2(
&batch_normalization_10/batchnorm/mul_1?
&batch_normalization_10/batchnorm/mul_2Mul/batch_normalization_10/moments/Squeeze:output:0(batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_10/batchnorm/mul_2?
$batch_normalization_10/batchnorm/subSub2batch_normalization_10/Cast/ReadVariableOp:value:0*batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2&
$batch_normalization_10/batchnorm/sub?
&batch_normalization_10/batchnorm/add_1AddV2*batch_normalization_10/batchnorm/mul_1:z:0(batch_normalization_10/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2(
&batch_normalization_10/batchnorm/add_1?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02 
dense_12/MatMul/ReadVariableOp?
dense_12/MatMulMatMul*batch_normalization_10/batchnorm/add_1:z:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_12/MatMul?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_12/BiasAdd/ReadVariableOp?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_12/BiasAdds
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_12/Relu?
4batch_normalization_9/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_9/moments/mean/reduction_indices?
"batch_normalization_9/moments/meanMeandense_12/Relu:activations:0=batch_normalization_9/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2$
"batch_normalization_9/moments/mean?
*batch_normalization_9/moments/StopGradientStopGradient+batch_normalization_9/moments/mean:output:0*
T0*
_output_shapes

:@2,
*batch_normalization_9/moments/StopGradient?
/batch_normalization_9/moments/SquaredDifferenceSquaredDifferencedense_12/Relu:activations:03batch_normalization_9/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????@21
/batch_normalization_9/moments/SquaredDifference?
8batch_normalization_9/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_9/moments/variance/reduction_indices?
&batch_normalization_9/moments/varianceMean3batch_normalization_9/moments/SquaredDifference:z:0Abatch_normalization_9/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2(
&batch_normalization_9/moments/variance?
%batch_normalization_9/moments/SqueezeSqueeze+batch_normalization_9/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2'
%batch_normalization_9/moments/Squeeze?
'batch_normalization_9/moments/Squeeze_1Squeeze/batch_normalization_9/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2)
'batch_normalization_9/moments/Squeeze_1?
+batch_normalization_9/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+batch_normalization_9/AssignMovingAvg/decay?
4batch_normalization_9/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_9_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype026
4batch_normalization_9/AssignMovingAvg/ReadVariableOp?
)batch_normalization_9/AssignMovingAvg/subSub<batch_normalization_9/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_9/moments/Squeeze:output:0*
T0*
_output_shapes
:@2+
)batch_normalization_9/AssignMovingAvg/sub?
)batch_normalization_9/AssignMovingAvg/mulMul-batch_normalization_9/AssignMovingAvg/sub:z:04batch_normalization_9/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2+
)batch_normalization_9/AssignMovingAvg/mul?
%batch_normalization_9/AssignMovingAvgAssignSubVariableOp=batch_normalization_9_assignmovingavg_readvariableop_resource-batch_normalization_9/AssignMovingAvg/mul:z:05^batch_normalization_9/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_9/AssignMovingAvg?
-batch_normalization_9/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-batch_normalization_9/AssignMovingAvg_1/decay?
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_9_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp?
+batch_normalization_9/AssignMovingAvg_1/subSub>batch_normalization_9/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_9/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2-
+batch_normalization_9/AssignMovingAvg_1/sub?
+batch_normalization_9/AssignMovingAvg_1/mulMul/batch_normalization_9/AssignMovingAvg_1/sub:z:06batch_normalization_9/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2-
+batch_normalization_9/AssignMovingAvg_1/mul?
'batch_normalization_9/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_9_assignmovingavg_1_readvariableop_resource/batch_normalization_9/AssignMovingAvg_1/mul:z:07^batch_normalization_9/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_9/AssignMovingAvg_1?
)batch_normalization_9/Cast/ReadVariableOpReadVariableOp2batch_normalization_9_cast_readvariableop_resource*
_output_shapes
:@*
dtype02+
)batch_normalization_9/Cast/ReadVariableOp?
+batch_normalization_9/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_9_cast_1_readvariableop_resource*
_output_shapes
:@*
dtype02-
+batch_normalization_9/Cast_1/ReadVariableOp?
%batch_normalization_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%batch_normalization_9/batchnorm/add/y?
#batch_normalization_9/batchnorm/addAddV20batch_normalization_9/moments/Squeeze_1:output:0.batch_normalization_9/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_9/batchnorm/add?
%batch_normalization_9/batchnorm/RsqrtRsqrt'batch_normalization_9/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_9/batchnorm/Rsqrt?
#batch_normalization_9/batchnorm/mulMul)batch_normalization_9/batchnorm/Rsqrt:y:03batch_normalization_9/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_9/batchnorm/mul?
%batch_normalization_9/batchnorm/mul_1Muldense_12/Relu:activations:0'batch_normalization_9/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????@2'
%batch_normalization_9/batchnorm/mul_1?
%batch_normalization_9/batchnorm/mul_2Mul.batch_normalization_9/moments/Squeeze:output:0'batch_normalization_9/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_9/batchnorm/mul_2?
#batch_normalization_9/batchnorm/subSub1batch_normalization_9/Cast/ReadVariableOp:value:0)batch_normalization_9/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_9/batchnorm/sub?
%batch_normalization_9/batchnorm/add_1AddV2)batch_normalization_9/batchnorm/mul_1:z:0'batch_normalization_9/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????@2'
%batch_normalization_9/batchnorm/add_1?
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02 
dense_11/MatMul/ReadVariableOp?
dense_11/MatMulMatMul)batch_normalization_9/batchnorm/add_1:z:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_11/MatMul?
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_11/BiasAdd/ReadVariableOp?
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_11/BiasAddt
dense_11/ReluReludense_11/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_11/Relu?
4batch_normalization_8/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_8/moments/mean/reduction_indices?
"batch_normalization_8/moments/meanMeandense_11/Relu:activations:0=batch_normalization_8/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2$
"batch_normalization_8/moments/mean?
*batch_normalization_8/moments/StopGradientStopGradient+batch_normalization_8/moments/mean:output:0*
T0*
_output_shapes
:	?2,
*batch_normalization_8/moments/StopGradient?
/batch_normalization_8/moments/SquaredDifferenceSquaredDifferencedense_11/Relu:activations:03batch_normalization_8/moments/StopGradient:output:0*
T0*(
_output_shapes
:??????????21
/batch_normalization_8/moments/SquaredDifference?
8batch_normalization_8/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_8/moments/variance/reduction_indices?
&batch_normalization_8/moments/varianceMean3batch_normalization_8/moments/SquaredDifference:z:0Abatch_normalization_8/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2(
&batch_normalization_8/moments/variance?
%batch_normalization_8/moments/SqueezeSqueeze+batch_normalization_8/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2'
%batch_normalization_8/moments/Squeeze?
'batch_normalization_8/moments/Squeeze_1Squeeze/batch_normalization_8/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2)
'batch_normalization_8/moments/Squeeze_1?
+batch_normalization_8/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+batch_normalization_8/AssignMovingAvg/decay?
4batch_normalization_8/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_8_assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype026
4batch_normalization_8/AssignMovingAvg/ReadVariableOp?
)batch_normalization_8/AssignMovingAvg/subSub<batch_normalization_8/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_8/moments/Squeeze:output:0*
T0*
_output_shapes	
:?2+
)batch_normalization_8/AssignMovingAvg/sub?
)batch_normalization_8/AssignMovingAvg/mulMul-batch_normalization_8/AssignMovingAvg/sub:z:04batch_normalization_8/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?2+
)batch_normalization_8/AssignMovingAvg/mul?
%batch_normalization_8/AssignMovingAvgAssignSubVariableOp=batch_normalization_8_assignmovingavg_readvariableop_resource-batch_normalization_8/AssignMovingAvg/mul:z:05^batch_normalization_8/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_8/AssignMovingAvg?
-batch_normalization_8/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-batch_normalization_8/AssignMovingAvg_1/decay?
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_8_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp?
+batch_normalization_8/AssignMovingAvg_1/subSub>batch_normalization_8/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_8/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?2-
+batch_normalization_8/AssignMovingAvg_1/sub?
+batch_normalization_8/AssignMovingAvg_1/mulMul/batch_normalization_8/AssignMovingAvg_1/sub:z:06batch_normalization_8/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?2-
+batch_normalization_8/AssignMovingAvg_1/mul?
'batch_normalization_8/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_8_assignmovingavg_1_readvariableop_resource/batch_normalization_8/AssignMovingAvg_1/mul:z:07^batch_normalization_8/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_8/AssignMovingAvg_1?
)batch_normalization_8/Cast/ReadVariableOpReadVariableOp2batch_normalization_8_cast_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)batch_normalization_8/Cast/ReadVariableOp?
+batch_normalization_8/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_8_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+batch_normalization_8/Cast_1/ReadVariableOp?
%batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%batch_normalization_8/batchnorm/add/y?
#batch_normalization_8/batchnorm/addAddV20batch_normalization_8/moments/Squeeze_1:output:0.batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2%
#batch_normalization_8/batchnorm/add?
%batch_normalization_8/batchnorm/RsqrtRsqrt'batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes	
:?2'
%batch_normalization_8/batchnorm/Rsqrt?
#batch_normalization_8/batchnorm/mulMul)batch_normalization_8/batchnorm/Rsqrt:y:03batch_normalization_8/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2%
#batch_normalization_8/batchnorm/mul?
%batch_normalization_8/batchnorm/mul_1Muldense_11/Relu:activations:0'batch_normalization_8/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2'
%batch_normalization_8/batchnorm/mul_1?
%batch_normalization_8/batchnorm/mul_2Mul.batch_normalization_8/moments/Squeeze:output:0'batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2'
%batch_normalization_8/batchnorm/mul_2?
#batch_normalization_8/batchnorm/subSub1batch_normalization_8/Cast/ReadVariableOp:value:0)batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2%
#batch_normalization_8/batchnorm/sub?
%batch_normalization_8/batchnorm/add_1AddV2)batch_normalization_8/batchnorm/mul_1:z:0'batch_normalization_8/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2'
%batch_normalization_8/batchnorm/add_1?
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes
:	?	*
dtype02 
dense_10/MatMul/ReadVariableOp?
dense_10/MatMulMatMul)batch_normalization_8/batchnorm/add_1:z:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
dense_10/MatMul?
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02!
dense_10/BiasAdd/ReadVariableOp?
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
dense_10/BiasAdd|
dense_10/SoftmaxSoftmaxdense_10/BiasAdd:output:0*
T0*'
_output_shapes
:?????????	2
dense_10/Softmax?
>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02@
>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp?
/sequential_1/dense_13/kernel/Regularizer/SquareSquareFsequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 ?21
/sequential_1/dense_13/kernel/Regularizer/Square?
.sequential_1/dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       20
.sequential_1/dense_13/kernel/Regularizer/Const?
,sequential_1/dense_13/kernel/Regularizer/SumSum3sequential_1/dense_13/kernel/Regularizer/Square:y:07sequential_1/dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2.
,sequential_1/dense_13/kernel/Regularizer/Sum?
.sequential_1/dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_1/dense_13/kernel/Regularizer/mul/x?
,sequential_1/dense_13/kernel/Regularizer/mulMul7sequential_1/dense_13/kernel/Regularizer/mul/x:output:05sequential_1/dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,sequential_1/dense_13/kernel/Regularizer/mul?
>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02@
>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp?
/sequential_1/dense_12/kernel/Regularizer/SquareSquareFsequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?@21
/sequential_1/dense_12/kernel/Regularizer/Square?
.sequential_1/dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       20
.sequential_1/dense_12/kernel/Regularizer/Const?
,sequential_1/dense_12/kernel/Regularizer/SumSum3sequential_1/dense_12/kernel/Regularizer/Square:y:07sequential_1/dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2.
,sequential_1/dense_12/kernel/Regularizer/Sum?
.sequential_1/dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>20
.sequential_1/dense_12/kernel/Regularizer/mul/x?
,sequential_1/dense_12/kernel/Regularizer/mulMul7sequential_1/dense_12/kernel/Regularizer/mul/x:output:05sequential_1/dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,sequential_1/dense_12/kernel/Regularizer/mul?
IdentityIdentitydense_10/Softmax:softmax:0'^batch_normalization_10/AssignMovingAvg6^batch_normalization_10/AssignMovingAvg/ReadVariableOp)^batch_normalization_10/AssignMovingAvg_18^batch_normalization_10/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_10/Cast/ReadVariableOp-^batch_normalization_10/Cast_1/ReadVariableOp'^batch_normalization_11/AssignMovingAvg6^batch_normalization_11/AssignMovingAvg/ReadVariableOp)^batch_normalization_11/AssignMovingAvg_18^batch_normalization_11/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_11/Cast/ReadVariableOp-^batch_normalization_11/Cast_1/ReadVariableOp&^batch_normalization_8/AssignMovingAvg5^batch_normalization_8/AssignMovingAvg/ReadVariableOp(^batch_normalization_8/AssignMovingAvg_17^batch_normalization_8/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_8/Cast/ReadVariableOp,^batch_normalization_8/Cast_1/ReadVariableOp&^batch_normalization_9/AssignMovingAvg5^batch_normalization_9/AssignMovingAvg/ReadVariableOp(^batch_normalization_9/AssignMovingAvg_17^batch_normalization_9/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_9/Cast/ReadVariableOp,^batch_normalization_9/Cast_1/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp?^sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp?^sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&batch_normalization_10/AssignMovingAvg&batch_normalization_10/AssignMovingAvg2n
5batch_normalization_10/AssignMovingAvg/ReadVariableOp5batch_normalization_10/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_10/AssignMovingAvg_1(batch_normalization_10/AssignMovingAvg_12r
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_10/Cast/ReadVariableOp*batch_normalization_10/Cast/ReadVariableOp2\
,batch_normalization_10/Cast_1/ReadVariableOp,batch_normalization_10/Cast_1/ReadVariableOp2P
&batch_normalization_11/AssignMovingAvg&batch_normalization_11/AssignMovingAvg2n
5batch_normalization_11/AssignMovingAvg/ReadVariableOp5batch_normalization_11/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_11/AssignMovingAvg_1(batch_normalization_11/AssignMovingAvg_12r
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_11/Cast/ReadVariableOp*batch_normalization_11/Cast/ReadVariableOp2\
,batch_normalization_11/Cast_1/ReadVariableOp,batch_normalization_11/Cast_1/ReadVariableOp2N
%batch_normalization_8/AssignMovingAvg%batch_normalization_8/AssignMovingAvg2l
4batch_normalization_8/AssignMovingAvg/ReadVariableOp4batch_normalization_8/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_8/AssignMovingAvg_1'batch_normalization_8/AssignMovingAvg_12p
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_8/Cast/ReadVariableOp)batch_normalization_8/Cast/ReadVariableOp2Z
+batch_normalization_8/Cast_1/ReadVariableOp+batch_normalization_8/Cast_1/ReadVariableOp2N
%batch_normalization_9/AssignMovingAvg%batch_normalization_9/AssignMovingAvg2l
4batch_normalization_9/AssignMovingAvg/ReadVariableOp4batch_normalization_9/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_9/AssignMovingAvg_1'batch_normalization_9/AssignMovingAvg_12p
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_9/Cast/ReadVariableOp)batch_normalization_9/Cast/ReadVariableOp2Z
+batch_normalization_9/Cast_1/ReadVariableOp+batch_normalization_9/Cast_1/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2?
>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp2?
>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp:` \
'
_output_shapes
:?????????
1
_user_specified_nameinputs/category_embed_1:`\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/category_embed_2:`\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/category_embed_3:`\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/category_embed_4:`\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/category_embed_5:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/city_embed_1:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/city_embed_2:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/city_embed_3:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/city_embed_4:\	X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/city_embed_5:U
Q
'
_output_shapes
:?????????
&
_user_specified_nameinputs/colon:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/commas:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/dash:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/exclam:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/money:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/month:[W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/parenthesis:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/state_embed_1:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/state_embed_2:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/state_embed_3:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/state_embed_4:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/state_embed_5:WS
'
_output_shapes
:?????????
(
_user_specified_nameinputs/weekday
?*
?

-__inference_sequential_1_layer_call_fn_562605
inputs_category_embed_1
inputs_category_embed_2
inputs_category_embed_3
inputs_category_embed_4
inputs_category_embed_5
inputs_city_embed_1
inputs_city_embed_2
inputs_city_embed_3
inputs_city_embed_4
inputs_city_embed_5
inputs_colon
inputs_commas
inputs_dash
inputs_exclam
inputs_money
inputs_month
inputs_parenthesis
inputs_state_embed_1
inputs_state_embed_2
inputs_state_embed_3
inputs_state_embed_4
inputs_state_embed_5
inputs_weekday
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:	 ?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:	?@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:	@?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?

unknown_23:	?	

unknown_24:	
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallinputs_category_embed_1inputs_category_embed_2inputs_category_embed_3inputs_category_embed_4inputs_category_embed_5inputs_city_embed_1inputs_city_embed_2inputs_city_embed_3inputs_city_embed_4inputs_city_embed_5inputs_coloninputs_commasinputs_dashinputs_exclaminputs_moneyinputs_monthinputs_parenthesisinputs_state_embed_1inputs_state_embed_2inputs_state_embed_3inputs_state_embed_4inputs_state_embed_5inputs_weekdayunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*<
_read_only_resource_inputs
 !"#$%&'()*+,-./0*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_5615182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
'
_output_shapes
:?????????
1
_user_specified_nameinputs/category_embed_1:`\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/category_embed_2:`\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/category_embed_3:`\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/category_embed_4:`\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/category_embed_5:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/city_embed_1:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/city_embed_2:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/city_embed_3:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/city_embed_4:\	X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/city_embed_5:U
Q
'
_output_shapes
:?????????
&
_user_specified_nameinputs/colon:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/commas:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/dash:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/exclam:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/money:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/month:[W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/parenthesis:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/state_embed_1:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/state_embed_2:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/state_embed_3:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/state_embed_4:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/state_embed_5:WS
'
_output_shapes
:?????????
(
_user_specified_nameinputs/weekday
?
?
D__inference_dense_13_layer_call_and_return_conditional_losses_561415

inputs1
matmul_readvariableop_resource:	 ?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02@
>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp?
/sequential_1/dense_13/kernel/Regularizer/SquareSquareFsequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 ?21
/sequential_1/dense_13/kernel/Regularizer/Square?
.sequential_1/dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       20
.sequential_1/dense_13/kernel/Regularizer/Const?
,sequential_1/dense_13/kernel/Regularizer/SumSum3sequential_1/dense_13/kernel/Regularizer/Square:y:07sequential_1/dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2.
,sequential_1/dense_13/kernel/Regularizer/Sum?
.sequential_1/dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_1/dense_13/kernel/Regularizer/mul/x?
,sequential_1/dense_13/kernel/Regularizer/mulMul7sequential_1/dense_13/kernel/Regularizer/mul/x:output:05sequential_1/dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,sequential_1/dense_13/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp?^sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2?
>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
??
?
L__inference_dense_features_1_layer_call_and_return_conditional_losses_561887
features

features_1

features_2

features_3

features_4

features_5

features_6

features_7

features_8

features_9
features_10
features_11
features_12
features_13
features_14
features_15
features_16
features_17
features_18
features_19
features_20
features_21
features_22
identityh
category_embed_1/ShapeShapefeatures*
T0*
_output_shapes
:2
category_embed_1/Shape?
$category_embed_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$category_embed_1/strided_slice/stack?
&category_embed_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_1/strided_slice/stack_1?
&category_embed_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_1/strided_slice/stack_2?
category_embed_1/strided_sliceStridedSlicecategory_embed_1/Shape:output:0-category_embed_1/strided_slice/stack:output:0/category_embed_1/strided_slice/stack_1:output:0/category_embed_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
category_embed_1/strided_slice?
 category_embed_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 category_embed_1/Reshape/shape/1?
category_embed_1/Reshape/shapePack'category_embed_1/strided_slice:output:0)category_embed_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2 
category_embed_1/Reshape/shape?
category_embed_1/ReshapeReshapefeatures'category_embed_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
category_embed_1/Reshapej
category_embed_2/ShapeShape
features_1*
T0*
_output_shapes
:2
category_embed_2/Shape?
$category_embed_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$category_embed_2/strided_slice/stack?
&category_embed_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_2/strided_slice/stack_1?
&category_embed_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_2/strided_slice/stack_2?
category_embed_2/strided_sliceStridedSlicecategory_embed_2/Shape:output:0-category_embed_2/strided_slice/stack:output:0/category_embed_2/strided_slice/stack_1:output:0/category_embed_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
category_embed_2/strided_slice?
 category_embed_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 category_embed_2/Reshape/shape/1?
category_embed_2/Reshape/shapePack'category_embed_2/strided_slice:output:0)category_embed_2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2 
category_embed_2/Reshape/shape?
category_embed_2/ReshapeReshape
features_1'category_embed_2/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
category_embed_2/Reshapej
category_embed_3/ShapeShape
features_2*
T0*
_output_shapes
:2
category_embed_3/Shape?
$category_embed_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$category_embed_3/strided_slice/stack?
&category_embed_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_3/strided_slice/stack_1?
&category_embed_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_3/strided_slice/stack_2?
category_embed_3/strided_sliceStridedSlicecategory_embed_3/Shape:output:0-category_embed_3/strided_slice/stack:output:0/category_embed_3/strided_slice/stack_1:output:0/category_embed_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
category_embed_3/strided_slice?
 category_embed_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 category_embed_3/Reshape/shape/1?
category_embed_3/Reshape/shapePack'category_embed_3/strided_slice:output:0)category_embed_3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2 
category_embed_3/Reshape/shape?
category_embed_3/ReshapeReshape
features_2'category_embed_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
category_embed_3/Reshapej
category_embed_4/ShapeShape
features_3*
T0*
_output_shapes
:2
category_embed_4/Shape?
$category_embed_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$category_embed_4/strided_slice/stack?
&category_embed_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_4/strided_slice/stack_1?
&category_embed_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_4/strided_slice/stack_2?
category_embed_4/strided_sliceStridedSlicecategory_embed_4/Shape:output:0-category_embed_4/strided_slice/stack:output:0/category_embed_4/strided_slice/stack_1:output:0/category_embed_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
category_embed_4/strided_slice?
 category_embed_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 category_embed_4/Reshape/shape/1?
category_embed_4/Reshape/shapePack'category_embed_4/strided_slice:output:0)category_embed_4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2 
category_embed_4/Reshape/shape?
category_embed_4/ReshapeReshape
features_3'category_embed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
category_embed_4/Reshapej
category_embed_5/ShapeShape
features_4*
T0*
_output_shapes
:2
category_embed_5/Shape?
$category_embed_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$category_embed_5/strided_slice/stack?
&category_embed_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_5/strided_slice/stack_1?
&category_embed_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_5/strided_slice/stack_2?
category_embed_5/strided_sliceStridedSlicecategory_embed_5/Shape:output:0-category_embed_5/strided_slice/stack:output:0/category_embed_5/strided_slice/stack_1:output:0/category_embed_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
category_embed_5/strided_slice?
 category_embed_5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 category_embed_5/Reshape/shape/1?
category_embed_5/Reshape/shapePack'category_embed_5/strided_slice:output:0)category_embed_5/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2 
category_embed_5/Reshape/shape?
category_embed_5/ReshapeReshape
features_4'category_embed_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
category_embed_5/Reshapeb
city_embed_1/ShapeShape
features_5*
T0*
_output_shapes
:2
city_embed_1/Shape?
 city_embed_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 city_embed_1/strided_slice/stack?
"city_embed_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_1/strided_slice/stack_1?
"city_embed_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_1/strided_slice/stack_2?
city_embed_1/strided_sliceStridedSlicecity_embed_1/Shape:output:0)city_embed_1/strided_slice/stack:output:0+city_embed_1/strided_slice/stack_1:output:0+city_embed_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
city_embed_1/strided_slice~
city_embed_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
city_embed_1/Reshape/shape/1?
city_embed_1/Reshape/shapePack#city_embed_1/strided_slice:output:0%city_embed_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
city_embed_1/Reshape/shape?
city_embed_1/ReshapeReshape
features_5#city_embed_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
city_embed_1/Reshapeb
city_embed_2/ShapeShape
features_6*
T0*
_output_shapes
:2
city_embed_2/Shape?
 city_embed_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 city_embed_2/strided_slice/stack?
"city_embed_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_2/strided_slice/stack_1?
"city_embed_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_2/strided_slice/stack_2?
city_embed_2/strided_sliceStridedSlicecity_embed_2/Shape:output:0)city_embed_2/strided_slice/stack:output:0+city_embed_2/strided_slice/stack_1:output:0+city_embed_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
city_embed_2/strided_slice~
city_embed_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
city_embed_2/Reshape/shape/1?
city_embed_2/Reshape/shapePack#city_embed_2/strided_slice:output:0%city_embed_2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
city_embed_2/Reshape/shape?
city_embed_2/ReshapeReshape
features_6#city_embed_2/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
city_embed_2/Reshapeb
city_embed_3/ShapeShape
features_7*
T0*
_output_shapes
:2
city_embed_3/Shape?
 city_embed_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 city_embed_3/strided_slice/stack?
"city_embed_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_3/strided_slice/stack_1?
"city_embed_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_3/strided_slice/stack_2?
city_embed_3/strided_sliceStridedSlicecity_embed_3/Shape:output:0)city_embed_3/strided_slice/stack:output:0+city_embed_3/strided_slice/stack_1:output:0+city_embed_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
city_embed_3/strided_slice~
city_embed_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
city_embed_3/Reshape/shape/1?
city_embed_3/Reshape/shapePack#city_embed_3/strided_slice:output:0%city_embed_3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
city_embed_3/Reshape/shape?
city_embed_3/ReshapeReshape
features_7#city_embed_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
city_embed_3/Reshapeb
city_embed_4/ShapeShape
features_8*
T0*
_output_shapes
:2
city_embed_4/Shape?
 city_embed_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 city_embed_4/strided_slice/stack?
"city_embed_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_4/strided_slice/stack_1?
"city_embed_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_4/strided_slice/stack_2?
city_embed_4/strided_sliceStridedSlicecity_embed_4/Shape:output:0)city_embed_4/strided_slice/stack:output:0+city_embed_4/strided_slice/stack_1:output:0+city_embed_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
city_embed_4/strided_slice~
city_embed_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
city_embed_4/Reshape/shape/1?
city_embed_4/Reshape/shapePack#city_embed_4/strided_slice:output:0%city_embed_4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
city_embed_4/Reshape/shape?
city_embed_4/ReshapeReshape
features_8#city_embed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
city_embed_4/Reshapeb
city_embed_5/ShapeShape
features_9*
T0*
_output_shapes
:2
city_embed_5/Shape?
 city_embed_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 city_embed_5/strided_slice/stack?
"city_embed_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_5/strided_slice/stack_1?
"city_embed_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_5/strided_slice/stack_2?
city_embed_5/strided_sliceStridedSlicecity_embed_5/Shape:output:0)city_embed_5/strided_slice/stack:output:0+city_embed_5/strided_slice/stack_1:output:0+city_embed_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
city_embed_5/strided_slice~
city_embed_5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
city_embed_5/Reshape/shape/1?
city_embed_5/Reshape/shapePack#city_embed_5/strided_slice:output:0%city_embed_5/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
city_embed_5/Reshape/shape?
city_embed_5/ReshapeReshape
features_9#city_embed_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
city_embed_5/ReshapeU
colon/ShapeShapefeatures_10*
T0*
_output_shapes
:2
colon/Shape?
colon/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
colon/strided_slice/stack?
colon/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
colon/strided_slice/stack_1?
colon/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
colon/strided_slice/stack_2?
colon/strided_sliceStridedSlicecolon/Shape:output:0"colon/strided_slice/stack:output:0$colon/strided_slice/stack_1:output:0$colon/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
colon/strided_slicep
colon/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
colon/Reshape/shape/1?
colon/Reshape/shapePackcolon/strided_slice:output:0colon/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
colon/Reshape/shape?
colon/ReshapeReshapefeatures_10colon/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
colon/ReshapeW
commas/ShapeShapefeatures_11*
T0*
_output_shapes
:2
commas/Shape?
commas/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
commas/strided_slice/stack?
commas/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
commas/strided_slice/stack_1?
commas/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
commas/strided_slice/stack_2?
commas/strided_sliceStridedSlicecommas/Shape:output:0#commas/strided_slice/stack:output:0%commas/strided_slice/stack_1:output:0%commas/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
commas/strided_slicer
commas/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
commas/Reshape/shape/1?
commas/Reshape/shapePackcommas/strided_slice:output:0commas/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
commas/Reshape/shape?
commas/ReshapeReshapefeatures_11commas/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
commas/ReshapeS

dash/ShapeShapefeatures_12*
T0*
_output_shapes
:2

dash/Shape~
dash/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
dash/strided_slice/stack?
dash/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
dash/strided_slice/stack_1?
dash/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
dash/strided_slice/stack_2?
dash/strided_sliceStridedSlicedash/Shape:output:0!dash/strided_slice/stack:output:0#dash/strided_slice/stack_1:output:0#dash/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dash/strided_slicen
dash/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
dash/Reshape/shape/1?
dash/Reshape/shapePackdash/strided_slice:output:0dash/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
dash/Reshape/shape?
dash/ReshapeReshapefeatures_12dash/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
dash/ReshapeW
exclam/ShapeShapefeatures_13*
T0*
_output_shapes
:2
exclam/Shape?
exclam/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
exclam/strided_slice/stack?
exclam/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
exclam/strided_slice/stack_1?
exclam/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
exclam/strided_slice/stack_2?
exclam/strided_sliceStridedSliceexclam/Shape:output:0#exclam/strided_slice/stack:output:0%exclam/strided_slice/stack_1:output:0%exclam/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
exclam/strided_slicer
exclam/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
exclam/Reshape/shape/1?
exclam/Reshape/shapePackexclam/strided_slice:output:0exclam/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
exclam/Reshape/shape?
exclam/ReshapeReshapefeatures_13exclam/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
exclam/ReshapeU
money/ShapeShapefeatures_14*
T0*
_output_shapes
:2
money/Shape?
money/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
money/strided_slice/stack?
money/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
money/strided_slice/stack_1?
money/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
money/strided_slice/stack_2?
money/strided_sliceStridedSlicemoney/Shape:output:0"money/strided_slice/stack:output:0$money/strided_slice/stack_1:output:0$money/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
money/strided_slicep
money/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
money/Reshape/shape/1?
money/Reshape/shapePackmoney/strided_slice:output:0money/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
money/Reshape/shape?
money/ReshapeReshapefeatures_14money/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
money/ReshapeU
month/ShapeShapefeatures_15*
T0*
_output_shapes
:2
month/Shape?
month/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
month/strided_slice/stack?
month/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
month/strided_slice/stack_1?
month/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
month/strided_slice/stack_2?
month/strided_sliceStridedSlicemonth/Shape:output:0"month/strided_slice/stack:output:0$month/strided_slice/stack_1:output:0$month/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
month/strided_slicep
month/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
month/Reshape/shape/1?
month/Reshape/shapePackmonth/strided_slice:output:0month/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
month/Reshape/shape?
month/ReshapeReshapefeatures_15month/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
month/Reshapea
parenthesis/ShapeShapefeatures_16*
T0*
_output_shapes
:2
parenthesis/Shape?
parenthesis/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
parenthesis/strided_slice/stack?
!parenthesis/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!parenthesis/strided_slice/stack_1?
!parenthesis/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!parenthesis/strided_slice/stack_2?
parenthesis/strided_sliceStridedSliceparenthesis/Shape:output:0(parenthesis/strided_slice/stack:output:0*parenthesis/strided_slice/stack_1:output:0*parenthesis/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
parenthesis/strided_slice|
parenthesis/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
parenthesis/Reshape/shape/1?
parenthesis/Reshape/shapePack"parenthesis/strided_slice:output:0$parenthesis/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
parenthesis/Reshape/shape?
parenthesis/ReshapeReshapefeatures_16"parenthesis/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
parenthesis/Reshapee
state_embed_1/ShapeShapefeatures_17*
T0*
_output_shapes
:2
state_embed_1/Shape?
!state_embed_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!state_embed_1/strided_slice/stack?
#state_embed_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_1/strided_slice/stack_1?
#state_embed_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_1/strided_slice/stack_2?
state_embed_1/strided_sliceStridedSlicestate_embed_1/Shape:output:0*state_embed_1/strided_slice/stack:output:0,state_embed_1/strided_slice/stack_1:output:0,state_embed_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
state_embed_1/strided_slice?
state_embed_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
state_embed_1/Reshape/shape/1?
state_embed_1/Reshape/shapePack$state_embed_1/strided_slice:output:0&state_embed_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
state_embed_1/Reshape/shape?
state_embed_1/ReshapeReshapefeatures_17$state_embed_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
state_embed_1/Reshapee
state_embed_2/ShapeShapefeatures_18*
T0*
_output_shapes
:2
state_embed_2/Shape?
!state_embed_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!state_embed_2/strided_slice/stack?
#state_embed_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_2/strided_slice/stack_1?
#state_embed_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_2/strided_slice/stack_2?
state_embed_2/strided_sliceStridedSlicestate_embed_2/Shape:output:0*state_embed_2/strided_slice/stack:output:0,state_embed_2/strided_slice/stack_1:output:0,state_embed_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
state_embed_2/strided_slice?
state_embed_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
state_embed_2/Reshape/shape/1?
state_embed_2/Reshape/shapePack$state_embed_2/strided_slice:output:0&state_embed_2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
state_embed_2/Reshape/shape?
state_embed_2/ReshapeReshapefeatures_18$state_embed_2/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
state_embed_2/Reshapee
state_embed_3/ShapeShapefeatures_19*
T0*
_output_shapes
:2
state_embed_3/Shape?
!state_embed_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!state_embed_3/strided_slice/stack?
#state_embed_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_3/strided_slice/stack_1?
#state_embed_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_3/strided_slice/stack_2?
state_embed_3/strided_sliceStridedSlicestate_embed_3/Shape:output:0*state_embed_3/strided_slice/stack:output:0,state_embed_3/strided_slice/stack_1:output:0,state_embed_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
state_embed_3/strided_slice?
state_embed_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
state_embed_3/Reshape/shape/1?
state_embed_3/Reshape/shapePack$state_embed_3/strided_slice:output:0&state_embed_3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
state_embed_3/Reshape/shape?
state_embed_3/ReshapeReshapefeatures_19$state_embed_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
state_embed_3/Reshapee
state_embed_4/ShapeShapefeatures_20*
T0*
_output_shapes
:2
state_embed_4/Shape?
!state_embed_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!state_embed_4/strided_slice/stack?
#state_embed_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_4/strided_slice/stack_1?
#state_embed_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_4/strided_slice/stack_2?
state_embed_4/strided_sliceStridedSlicestate_embed_4/Shape:output:0*state_embed_4/strided_slice/stack:output:0,state_embed_4/strided_slice/stack_1:output:0,state_embed_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
state_embed_4/strided_slice?
state_embed_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
state_embed_4/Reshape/shape/1?
state_embed_4/Reshape/shapePack$state_embed_4/strided_slice:output:0&state_embed_4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
state_embed_4/Reshape/shape?
state_embed_4/ReshapeReshapefeatures_20$state_embed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
state_embed_4/Reshapee
state_embed_5/ShapeShapefeatures_21*
T0*
_output_shapes
:2
state_embed_5/Shape?
!state_embed_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!state_embed_5/strided_slice/stack?
#state_embed_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_5/strided_slice/stack_1?
#state_embed_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_5/strided_slice/stack_2?
state_embed_5/strided_sliceStridedSlicestate_embed_5/Shape:output:0*state_embed_5/strided_slice/stack:output:0,state_embed_5/strided_slice/stack_1:output:0,state_embed_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
state_embed_5/strided_slice?
state_embed_5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
state_embed_5/Reshape/shape/1?
state_embed_5/Reshape/shapePack$state_embed_5/strided_slice:output:0&state_embed_5/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
state_embed_5/Reshape/shape?
state_embed_5/ReshapeReshapefeatures_21$state_embed_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
state_embed_5/ReshapeY
weekday/ShapeShapefeatures_22*
T0*
_output_shapes
:2
weekday/Shape?
weekday/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
weekday/strided_slice/stack?
weekday/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
weekday/strided_slice/stack_1?
weekday/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
weekday/strided_slice/stack_2?
weekday/strided_sliceStridedSliceweekday/Shape:output:0$weekday/strided_slice/stack:output:0&weekday/strided_slice/stack_1:output:0&weekday/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
weekday/strided_slicet
weekday/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
weekday/Reshape/shape/1?
weekday/Reshape/shapePackweekday/strided_slice:output:0 weekday/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
weekday/Reshape/shape?
weekday/ReshapeReshapefeatures_22weekday/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
weekday/Reshapee
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2!category_embed_1/Reshape:output:0!category_embed_2/Reshape:output:0!category_embed_3/Reshape:output:0!category_embed_4/Reshape:output:0!category_embed_5/Reshape:output:0city_embed_1/Reshape:output:0city_embed_2/Reshape:output:0city_embed_3/Reshape:output:0city_embed_4/Reshape:output:0city_embed_5/Reshape:output:0colon/Reshape:output:0commas/Reshape:output:0dash/Reshape:output:0exclam/Reshape:output:0money/Reshape:output:0month/Reshape:output:0parenthesis/Reshape:output:0state_embed_1/Reshape:output:0state_embed_2/Reshape:output:0state_embed_3/Reshape:output:0state_embed_4/Reshape:output:0state_embed_5/Reshape:output:0weekday/Reshape:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:Q	M
'
_output_shapes
:?????????
"
_user_specified_name
features:Q
M
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features:QM
'
_output_shapes
:?????????
"
_user_specified_name
features
?)
?
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_564188

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@*
cast_readvariableop_resource:@,
cast_1_readvariableop_resource:@
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????@2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:@*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:@*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????@2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?&
?	
-__inference_sequential_1_layer_call_fn_561573
category_embed_1
category_embed_2
category_embed_3
category_embed_4
category_embed_5
city_embed_1
city_embed_2
city_embed_3
city_embed_4
city_embed_5	
colon

commas
dash

exclam	
money	
month
parenthesis
state_embed_1
state_embed_2
state_embed_3
state_embed_4
state_embed_5
weekday
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:	 ?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:	?@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:	@?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?

unknown_23:	?	

unknown_24:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallcategory_embed_1category_embed_2category_embed_3category_embed_4category_embed_5city_embed_1city_embed_2city_embed_3city_embed_4city_embed_5coloncommasdashexclammoneymonthparenthesisstate_embed_1state_embed_2state_embed_3state_embed_4state_embed_5weekdayunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*<
_read_only_resource_inputs
 !"#$%&'()*+,-./0*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_5615182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
'
_output_shapes
:?????????
*
_user_specified_namecategory_embed_1:YU
'
_output_shapes
:?????????
*
_user_specified_namecategory_embed_2:YU
'
_output_shapes
:?????????
*
_user_specified_namecategory_embed_3:YU
'
_output_shapes
:?????????
*
_user_specified_namecategory_embed_4:YU
'
_output_shapes
:?????????
*
_user_specified_namecategory_embed_5:UQ
'
_output_shapes
:?????????
&
_user_specified_namecity_embed_1:UQ
'
_output_shapes
:?????????
&
_user_specified_namecity_embed_2:UQ
'
_output_shapes
:?????????
&
_user_specified_namecity_embed_3:UQ
'
_output_shapes
:?????????
&
_user_specified_namecity_embed_4:U	Q
'
_output_shapes
:?????????
&
_user_specified_namecity_embed_5:N
J
'
_output_shapes
:?????????

_user_specified_namecolon:OK
'
_output_shapes
:?????????
 
_user_specified_namecommas:MI
'
_output_shapes
:?????????

_user_specified_namedash:OK
'
_output_shapes
:?????????
 
_user_specified_nameexclam:NJ
'
_output_shapes
:?????????

_user_specified_namemoney:NJ
'
_output_shapes
:?????????

_user_specified_namemonth:TP
'
_output_shapes
:?????????
%
_user_specified_nameparenthesis:VR
'
_output_shapes
:?????????
'
_user_specified_namestate_embed_1:VR
'
_output_shapes
:?????????
'
_user_specified_namestate_embed_2:VR
'
_output_shapes
:?????????
'
_user_specified_namestate_embed_3:VR
'
_output_shapes
:?????????
'
_user_specified_namestate_embed_4:VR
'
_output_shapes
:?????????
'
_user_specified_namestate_embed_5:PL
'
_output_shapes
:?????????
!
_user_specified_name	weekday
?)
?
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_560543

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: *
cast_readvariableop_resource: ,
cast_1_readvariableop_resource: 
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

: 2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:????????? 2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
: *
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
: *
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:????????? 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:????????? 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?)
?
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_560705

inputs6
'assignmovingavg_readvariableop_resource:	?8
)assignmovingavg_1_readvariableop_resource:	?+
cast_readvariableop_resource:	?-
cast_1_readvariableop_resource:	?
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?"
?
1__inference_dense_features_1_layer_call_fn_563440
features_category_embed_1
features_category_embed_2
features_category_embed_3
features_category_embed_4
features_category_embed_5
features_city_embed_1
features_city_embed_2
features_city_embed_3
features_city_embed_4
features_city_embed_5
features_colon
features_commas
features_dash
features_exclam
features_money
features_month
features_parenthesis
features_state_embed_1
features_state_embed_2
features_state_embed_3
features_state_embed_4
features_state_embed_5
features_weekday
identity?
PartitionedCallPartitionedCallfeatures_category_embed_1features_category_embed_2features_category_embed_3features_category_embed_4features_category_embed_5features_city_embed_1features_city_embed_2features_city_embed_3features_city_embed_4features_city_embed_5features_colonfeatures_commasfeatures_dashfeatures_exclamfeatures_moneyfeatures_monthfeatures_parenthesisfeatures_state_embed_1features_state_embed_2features_state_embed_3features_state_embed_4features_state_embed_5features_weekday*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_dense_features_1_layer_call_and_return_conditional_losses_5618872
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:b ^
'
_output_shapes
:?????????
3
_user_specified_namefeatures/category_embed_1:b^
'
_output_shapes
:?????????
3
_user_specified_namefeatures/category_embed_2:b^
'
_output_shapes
:?????????
3
_user_specified_namefeatures/category_embed_3:b^
'
_output_shapes
:?????????
3
_user_specified_namefeatures/category_embed_4:b^
'
_output_shapes
:?????????
3
_user_specified_namefeatures/category_embed_5:^Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/city_embed_1:^Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/city_embed_2:^Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/city_embed_3:^Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/city_embed_4:^	Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/city_embed_5:W
S
'
_output_shapes
:?????????
(
_user_specified_namefeatures/colon:XT
'
_output_shapes
:?????????
)
_user_specified_namefeatures/commas:VR
'
_output_shapes
:?????????
'
_user_specified_namefeatures/dash:XT
'
_output_shapes
:?????????
)
_user_specified_namefeatures/exclam:WS
'
_output_shapes
:?????????
(
_user_specified_namefeatures/money:WS
'
_output_shapes
:?????????
(
_user_specified_namefeatures/month:]Y
'
_output_shapes
:?????????
.
_user_specified_namefeatures/parenthesis:_[
'
_output_shapes
:?????????
0
_user_specified_namefeatures/state_embed_1:_[
'
_output_shapes
:?????????
0
_user_specified_namefeatures/state_embed_2:_[
'
_output_shapes
:?????????
0
_user_specified_namefeatures/state_embed_3:_[
'
_output_shapes
:?????????
0
_user_specified_namefeatures/state_embed_4:_[
'
_output_shapes
:?????????
0
_user_specified_namefeatures/state_embed_5:YU
'
_output_shapes
:?????????
*
_user_specified_namefeatures/weekday
?

?
D__inference_dense_11_layer_call_and_return_conditional_losses_561473

inputs1
matmul_readvariableop_resource:	@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?
L__inference_dense_features_1_layer_call_and_return_conditional_losses_563864
features_category_embed_1
features_category_embed_2
features_category_embed_3
features_category_embed_4
features_category_embed_5
features_city_embed_1
features_city_embed_2
features_city_embed_3
features_city_embed_4
features_city_embed_5
features_colon
features_commas
features_dash
features_exclam
features_money
features_month
features_parenthesis
features_state_embed_1
features_state_embed_2
features_state_embed_3
features_state_embed_4
features_state_embed_5
features_weekday
identityy
category_embed_1/ShapeShapefeatures_category_embed_1*
T0*
_output_shapes
:2
category_embed_1/Shape?
$category_embed_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$category_embed_1/strided_slice/stack?
&category_embed_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_1/strided_slice/stack_1?
&category_embed_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_1/strided_slice/stack_2?
category_embed_1/strided_sliceStridedSlicecategory_embed_1/Shape:output:0-category_embed_1/strided_slice/stack:output:0/category_embed_1/strided_slice/stack_1:output:0/category_embed_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
category_embed_1/strided_slice?
 category_embed_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 category_embed_1/Reshape/shape/1?
category_embed_1/Reshape/shapePack'category_embed_1/strided_slice:output:0)category_embed_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2 
category_embed_1/Reshape/shape?
category_embed_1/ReshapeReshapefeatures_category_embed_1'category_embed_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
category_embed_1/Reshapey
category_embed_2/ShapeShapefeatures_category_embed_2*
T0*
_output_shapes
:2
category_embed_2/Shape?
$category_embed_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$category_embed_2/strided_slice/stack?
&category_embed_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_2/strided_slice/stack_1?
&category_embed_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_2/strided_slice/stack_2?
category_embed_2/strided_sliceStridedSlicecategory_embed_2/Shape:output:0-category_embed_2/strided_slice/stack:output:0/category_embed_2/strided_slice/stack_1:output:0/category_embed_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
category_embed_2/strided_slice?
 category_embed_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 category_embed_2/Reshape/shape/1?
category_embed_2/Reshape/shapePack'category_embed_2/strided_slice:output:0)category_embed_2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2 
category_embed_2/Reshape/shape?
category_embed_2/ReshapeReshapefeatures_category_embed_2'category_embed_2/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
category_embed_2/Reshapey
category_embed_3/ShapeShapefeatures_category_embed_3*
T0*
_output_shapes
:2
category_embed_3/Shape?
$category_embed_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$category_embed_3/strided_slice/stack?
&category_embed_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_3/strided_slice/stack_1?
&category_embed_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_3/strided_slice/stack_2?
category_embed_3/strided_sliceStridedSlicecategory_embed_3/Shape:output:0-category_embed_3/strided_slice/stack:output:0/category_embed_3/strided_slice/stack_1:output:0/category_embed_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
category_embed_3/strided_slice?
 category_embed_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 category_embed_3/Reshape/shape/1?
category_embed_3/Reshape/shapePack'category_embed_3/strided_slice:output:0)category_embed_3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2 
category_embed_3/Reshape/shape?
category_embed_3/ReshapeReshapefeatures_category_embed_3'category_embed_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
category_embed_3/Reshapey
category_embed_4/ShapeShapefeatures_category_embed_4*
T0*
_output_shapes
:2
category_embed_4/Shape?
$category_embed_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$category_embed_4/strided_slice/stack?
&category_embed_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_4/strided_slice/stack_1?
&category_embed_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_4/strided_slice/stack_2?
category_embed_4/strided_sliceStridedSlicecategory_embed_4/Shape:output:0-category_embed_4/strided_slice/stack:output:0/category_embed_4/strided_slice/stack_1:output:0/category_embed_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
category_embed_4/strided_slice?
 category_embed_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 category_embed_4/Reshape/shape/1?
category_embed_4/Reshape/shapePack'category_embed_4/strided_slice:output:0)category_embed_4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2 
category_embed_4/Reshape/shape?
category_embed_4/ReshapeReshapefeatures_category_embed_4'category_embed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
category_embed_4/Reshapey
category_embed_5/ShapeShapefeatures_category_embed_5*
T0*
_output_shapes
:2
category_embed_5/Shape?
$category_embed_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$category_embed_5/strided_slice/stack?
&category_embed_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_5/strided_slice/stack_1?
&category_embed_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&category_embed_5/strided_slice/stack_2?
category_embed_5/strided_sliceStridedSlicecategory_embed_5/Shape:output:0-category_embed_5/strided_slice/stack:output:0/category_embed_5/strided_slice/stack_1:output:0/category_embed_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
category_embed_5/strided_slice?
 category_embed_5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 category_embed_5/Reshape/shape/1?
category_embed_5/Reshape/shapePack'category_embed_5/strided_slice:output:0)category_embed_5/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2 
category_embed_5/Reshape/shape?
category_embed_5/ReshapeReshapefeatures_category_embed_5'category_embed_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
category_embed_5/Reshapem
city_embed_1/ShapeShapefeatures_city_embed_1*
T0*
_output_shapes
:2
city_embed_1/Shape?
 city_embed_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 city_embed_1/strided_slice/stack?
"city_embed_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_1/strided_slice/stack_1?
"city_embed_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_1/strided_slice/stack_2?
city_embed_1/strided_sliceStridedSlicecity_embed_1/Shape:output:0)city_embed_1/strided_slice/stack:output:0+city_embed_1/strided_slice/stack_1:output:0+city_embed_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
city_embed_1/strided_slice~
city_embed_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
city_embed_1/Reshape/shape/1?
city_embed_1/Reshape/shapePack#city_embed_1/strided_slice:output:0%city_embed_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
city_embed_1/Reshape/shape?
city_embed_1/ReshapeReshapefeatures_city_embed_1#city_embed_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
city_embed_1/Reshapem
city_embed_2/ShapeShapefeatures_city_embed_2*
T0*
_output_shapes
:2
city_embed_2/Shape?
 city_embed_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 city_embed_2/strided_slice/stack?
"city_embed_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_2/strided_slice/stack_1?
"city_embed_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_2/strided_slice/stack_2?
city_embed_2/strided_sliceStridedSlicecity_embed_2/Shape:output:0)city_embed_2/strided_slice/stack:output:0+city_embed_2/strided_slice/stack_1:output:0+city_embed_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
city_embed_2/strided_slice~
city_embed_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
city_embed_2/Reshape/shape/1?
city_embed_2/Reshape/shapePack#city_embed_2/strided_slice:output:0%city_embed_2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
city_embed_2/Reshape/shape?
city_embed_2/ReshapeReshapefeatures_city_embed_2#city_embed_2/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
city_embed_2/Reshapem
city_embed_3/ShapeShapefeatures_city_embed_3*
T0*
_output_shapes
:2
city_embed_3/Shape?
 city_embed_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 city_embed_3/strided_slice/stack?
"city_embed_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_3/strided_slice/stack_1?
"city_embed_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_3/strided_slice/stack_2?
city_embed_3/strided_sliceStridedSlicecity_embed_3/Shape:output:0)city_embed_3/strided_slice/stack:output:0+city_embed_3/strided_slice/stack_1:output:0+city_embed_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
city_embed_3/strided_slice~
city_embed_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
city_embed_3/Reshape/shape/1?
city_embed_3/Reshape/shapePack#city_embed_3/strided_slice:output:0%city_embed_3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
city_embed_3/Reshape/shape?
city_embed_3/ReshapeReshapefeatures_city_embed_3#city_embed_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
city_embed_3/Reshapem
city_embed_4/ShapeShapefeatures_city_embed_4*
T0*
_output_shapes
:2
city_embed_4/Shape?
 city_embed_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 city_embed_4/strided_slice/stack?
"city_embed_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_4/strided_slice/stack_1?
"city_embed_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_4/strided_slice/stack_2?
city_embed_4/strided_sliceStridedSlicecity_embed_4/Shape:output:0)city_embed_4/strided_slice/stack:output:0+city_embed_4/strided_slice/stack_1:output:0+city_embed_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
city_embed_4/strided_slice~
city_embed_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
city_embed_4/Reshape/shape/1?
city_embed_4/Reshape/shapePack#city_embed_4/strided_slice:output:0%city_embed_4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
city_embed_4/Reshape/shape?
city_embed_4/ReshapeReshapefeatures_city_embed_4#city_embed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
city_embed_4/Reshapem
city_embed_5/ShapeShapefeatures_city_embed_5*
T0*
_output_shapes
:2
city_embed_5/Shape?
 city_embed_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 city_embed_5/strided_slice/stack?
"city_embed_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_5/strided_slice/stack_1?
"city_embed_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"city_embed_5/strided_slice/stack_2?
city_embed_5/strided_sliceStridedSlicecity_embed_5/Shape:output:0)city_embed_5/strided_slice/stack:output:0+city_embed_5/strided_slice/stack_1:output:0+city_embed_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
city_embed_5/strided_slice~
city_embed_5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
city_embed_5/Reshape/shape/1?
city_embed_5/Reshape/shapePack#city_embed_5/strided_slice:output:0%city_embed_5/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
city_embed_5/Reshape/shape?
city_embed_5/ReshapeReshapefeatures_city_embed_5#city_embed_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
city_embed_5/ReshapeX
colon/ShapeShapefeatures_colon*
T0*
_output_shapes
:2
colon/Shape?
colon/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
colon/strided_slice/stack?
colon/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
colon/strided_slice/stack_1?
colon/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
colon/strided_slice/stack_2?
colon/strided_sliceStridedSlicecolon/Shape:output:0"colon/strided_slice/stack:output:0$colon/strided_slice/stack_1:output:0$colon/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
colon/strided_slicep
colon/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
colon/Reshape/shape/1?
colon/Reshape/shapePackcolon/strided_slice:output:0colon/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
colon/Reshape/shape?
colon/ReshapeReshapefeatures_coloncolon/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
colon/Reshape[
commas/ShapeShapefeatures_commas*
T0*
_output_shapes
:2
commas/Shape?
commas/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
commas/strided_slice/stack?
commas/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
commas/strided_slice/stack_1?
commas/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
commas/strided_slice/stack_2?
commas/strided_sliceStridedSlicecommas/Shape:output:0#commas/strided_slice/stack:output:0%commas/strided_slice/stack_1:output:0%commas/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
commas/strided_slicer
commas/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
commas/Reshape/shape/1?
commas/Reshape/shapePackcommas/strided_slice:output:0commas/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
commas/Reshape/shape?
commas/ReshapeReshapefeatures_commascommas/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
commas/ReshapeU

dash/ShapeShapefeatures_dash*
T0*
_output_shapes
:2

dash/Shape~
dash/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
dash/strided_slice/stack?
dash/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
dash/strided_slice/stack_1?
dash/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
dash/strided_slice/stack_2?
dash/strided_sliceStridedSlicedash/Shape:output:0!dash/strided_slice/stack:output:0#dash/strided_slice/stack_1:output:0#dash/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
dash/strided_slicen
dash/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
dash/Reshape/shape/1?
dash/Reshape/shapePackdash/strided_slice:output:0dash/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
dash/Reshape/shape?
dash/ReshapeReshapefeatures_dashdash/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
dash/Reshape[
exclam/ShapeShapefeatures_exclam*
T0*
_output_shapes
:2
exclam/Shape?
exclam/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
exclam/strided_slice/stack?
exclam/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
exclam/strided_slice/stack_1?
exclam/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
exclam/strided_slice/stack_2?
exclam/strided_sliceStridedSliceexclam/Shape:output:0#exclam/strided_slice/stack:output:0%exclam/strided_slice/stack_1:output:0%exclam/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
exclam/strided_slicer
exclam/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
exclam/Reshape/shape/1?
exclam/Reshape/shapePackexclam/strided_slice:output:0exclam/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
exclam/Reshape/shape?
exclam/ReshapeReshapefeatures_exclamexclam/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
exclam/ReshapeX
money/ShapeShapefeatures_money*
T0*
_output_shapes
:2
money/Shape?
money/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
money/strided_slice/stack?
money/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
money/strided_slice/stack_1?
money/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
money/strided_slice/stack_2?
money/strided_sliceStridedSlicemoney/Shape:output:0"money/strided_slice/stack:output:0$money/strided_slice/stack_1:output:0$money/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
money/strided_slicep
money/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
money/Reshape/shape/1?
money/Reshape/shapePackmoney/strided_slice:output:0money/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
money/Reshape/shape?
money/ReshapeReshapefeatures_moneymoney/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
money/ReshapeX
month/ShapeShapefeatures_month*
T0*
_output_shapes
:2
month/Shape?
month/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
month/strided_slice/stack?
month/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
month/strided_slice/stack_1?
month/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
month/strided_slice/stack_2?
month/strided_sliceStridedSlicemonth/Shape:output:0"month/strided_slice/stack:output:0$month/strided_slice/stack_1:output:0$month/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
month/strided_slicep
month/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
month/Reshape/shape/1?
month/Reshape/shapePackmonth/strided_slice:output:0month/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
month/Reshape/shape?
month/ReshapeReshapefeatures_monthmonth/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
month/Reshapej
parenthesis/ShapeShapefeatures_parenthesis*
T0*
_output_shapes
:2
parenthesis/Shape?
parenthesis/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
parenthesis/strided_slice/stack?
!parenthesis/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!parenthesis/strided_slice/stack_1?
!parenthesis/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!parenthesis/strided_slice/stack_2?
parenthesis/strided_sliceStridedSliceparenthesis/Shape:output:0(parenthesis/strided_slice/stack:output:0*parenthesis/strided_slice/stack_1:output:0*parenthesis/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
parenthesis/strided_slice|
parenthesis/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
parenthesis/Reshape/shape/1?
parenthesis/Reshape/shapePack"parenthesis/strided_slice:output:0$parenthesis/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
parenthesis/Reshape/shape?
parenthesis/ReshapeReshapefeatures_parenthesis"parenthesis/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
parenthesis/Reshapep
state_embed_1/ShapeShapefeatures_state_embed_1*
T0*
_output_shapes
:2
state_embed_1/Shape?
!state_embed_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!state_embed_1/strided_slice/stack?
#state_embed_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_1/strided_slice/stack_1?
#state_embed_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_1/strided_slice/stack_2?
state_embed_1/strided_sliceStridedSlicestate_embed_1/Shape:output:0*state_embed_1/strided_slice/stack:output:0,state_embed_1/strided_slice/stack_1:output:0,state_embed_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
state_embed_1/strided_slice?
state_embed_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
state_embed_1/Reshape/shape/1?
state_embed_1/Reshape/shapePack$state_embed_1/strided_slice:output:0&state_embed_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
state_embed_1/Reshape/shape?
state_embed_1/ReshapeReshapefeatures_state_embed_1$state_embed_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
state_embed_1/Reshapep
state_embed_2/ShapeShapefeatures_state_embed_2*
T0*
_output_shapes
:2
state_embed_2/Shape?
!state_embed_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!state_embed_2/strided_slice/stack?
#state_embed_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_2/strided_slice/stack_1?
#state_embed_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_2/strided_slice/stack_2?
state_embed_2/strided_sliceStridedSlicestate_embed_2/Shape:output:0*state_embed_2/strided_slice/stack:output:0,state_embed_2/strided_slice/stack_1:output:0,state_embed_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
state_embed_2/strided_slice?
state_embed_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
state_embed_2/Reshape/shape/1?
state_embed_2/Reshape/shapePack$state_embed_2/strided_slice:output:0&state_embed_2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
state_embed_2/Reshape/shape?
state_embed_2/ReshapeReshapefeatures_state_embed_2$state_embed_2/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
state_embed_2/Reshapep
state_embed_3/ShapeShapefeatures_state_embed_3*
T0*
_output_shapes
:2
state_embed_3/Shape?
!state_embed_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!state_embed_3/strided_slice/stack?
#state_embed_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_3/strided_slice/stack_1?
#state_embed_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_3/strided_slice/stack_2?
state_embed_3/strided_sliceStridedSlicestate_embed_3/Shape:output:0*state_embed_3/strided_slice/stack:output:0,state_embed_3/strided_slice/stack_1:output:0,state_embed_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
state_embed_3/strided_slice?
state_embed_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
state_embed_3/Reshape/shape/1?
state_embed_3/Reshape/shapePack$state_embed_3/strided_slice:output:0&state_embed_3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
state_embed_3/Reshape/shape?
state_embed_3/ReshapeReshapefeatures_state_embed_3$state_embed_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
state_embed_3/Reshapep
state_embed_4/ShapeShapefeatures_state_embed_4*
T0*
_output_shapes
:2
state_embed_4/Shape?
!state_embed_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!state_embed_4/strided_slice/stack?
#state_embed_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_4/strided_slice/stack_1?
#state_embed_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_4/strided_slice/stack_2?
state_embed_4/strided_sliceStridedSlicestate_embed_4/Shape:output:0*state_embed_4/strided_slice/stack:output:0,state_embed_4/strided_slice/stack_1:output:0,state_embed_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
state_embed_4/strided_slice?
state_embed_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
state_embed_4/Reshape/shape/1?
state_embed_4/Reshape/shapePack$state_embed_4/strided_slice:output:0&state_embed_4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
state_embed_4/Reshape/shape?
state_embed_4/ReshapeReshapefeatures_state_embed_4$state_embed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
state_embed_4/Reshapep
state_embed_5/ShapeShapefeatures_state_embed_5*
T0*
_output_shapes
:2
state_embed_5/Shape?
!state_embed_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!state_embed_5/strided_slice/stack?
#state_embed_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_5/strided_slice/stack_1?
#state_embed_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#state_embed_5/strided_slice/stack_2?
state_embed_5/strided_sliceStridedSlicestate_embed_5/Shape:output:0*state_embed_5/strided_slice/stack:output:0,state_embed_5/strided_slice/stack_1:output:0,state_embed_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
state_embed_5/strided_slice?
state_embed_5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
state_embed_5/Reshape/shape/1?
state_embed_5/Reshape/shapePack$state_embed_5/strided_slice:output:0&state_embed_5/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
state_embed_5/Reshape/shape?
state_embed_5/ReshapeReshapefeatures_state_embed_5$state_embed_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
state_embed_5/Reshape^
weekday/ShapeShapefeatures_weekday*
T0*
_output_shapes
:2
weekday/Shape?
weekday/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
weekday/strided_slice/stack?
weekday/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
weekday/strided_slice/stack_1?
weekday/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
weekday/strided_slice/stack_2?
weekday/strided_sliceStridedSliceweekday/Shape:output:0$weekday/strided_slice/stack:output:0&weekday/strided_slice/stack_1:output:0&weekday/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
weekday/strided_slicet
weekday/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
weekday/Reshape/shape/1?
weekday/Reshape/shapePackweekday/strided_slice:output:0 weekday/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
weekday/Reshape/shape?
weekday/ReshapeReshapefeatures_weekdayweekday/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
weekday/Reshapee
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2!category_embed_1/Reshape:output:0!category_embed_2/Reshape:output:0!category_embed_3/Reshape:output:0!category_embed_4/Reshape:output:0!category_embed_5/Reshape:output:0city_embed_1/Reshape:output:0city_embed_2/Reshape:output:0city_embed_3/Reshape:output:0city_embed_4/Reshape:output:0city_embed_5/Reshape:output:0colon/Reshape:output:0commas/Reshape:output:0dash/Reshape:output:0exclam/Reshape:output:0money/Reshape:output:0month/Reshape:output:0parenthesis/Reshape:output:0state_embed_1/Reshape:output:0state_embed_2/Reshape:output:0state_embed_3/Reshape:output:0state_embed_4/Reshape:output:0state_embed_5/Reshape:output:0weekday/Reshape:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:b ^
'
_output_shapes
:?????????
3
_user_specified_namefeatures/category_embed_1:b^
'
_output_shapes
:?????????
3
_user_specified_namefeatures/category_embed_2:b^
'
_output_shapes
:?????????
3
_user_specified_namefeatures/category_embed_3:b^
'
_output_shapes
:?????????
3
_user_specified_namefeatures/category_embed_4:b^
'
_output_shapes
:?????????
3
_user_specified_namefeatures/category_embed_5:^Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/city_embed_1:^Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/city_embed_2:^Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/city_embed_3:^Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/city_embed_4:^	Z
'
_output_shapes
:?????????
/
_user_specified_namefeatures/city_embed_5:W
S
'
_output_shapes
:?????????
(
_user_specified_namefeatures/colon:XT
'
_output_shapes
:?????????
)
_user_specified_namefeatures/commas:VR
'
_output_shapes
:?????????
'
_user_specified_namefeatures/dash:XT
'
_output_shapes
:?????????
)
_user_specified_namefeatures/exclam:WS
'
_output_shapes
:?????????
(
_user_specified_namefeatures/money:WS
'
_output_shapes
:?????????
(
_user_specified_namefeatures/month:]Y
'
_output_shapes
:?????????
.
_user_specified_namefeatures/parenthesis:_[
'
_output_shapes
:?????????
0
_user_specified_namefeatures/state_embed_1:_[
'
_output_shapes
:?????????
0
_user_specified_namefeatures/state_embed_2:_[
'
_output_shapes
:?????????
0
_user_specified_namefeatures/state_embed_3:_[
'
_output_shapes
:?????????
0
_user_specified_namefeatures/state_embed_4:_[
'
_output_shapes
:?????????
0
_user_specified_namefeatures/state_embed_5:YU
'
_output_shapes
:?????????
*
_user_specified_namefeatures/weekday
?
?
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_563930

inputs*
cast_readvariableop_resource: ,
cast_1_readvariableop_resource: ,
cast_2_readvariableop_resource: ,
cast_3_readvariableop_resource: 
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
: *
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
: *
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
: *
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
: *
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:????????? 2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:????????? 2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_564042

inputs+
cast_readvariableop_resource:	?-
cast_1_readvariableop_resource:	?-
cast_2_readvariableop_resource:	?-
cast_3_readvariableop_resource:	?
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_9_layer_call_fn_564121

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_5608072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
)__inference_dense_11_layer_call_fn_564197

inputs
unknown:	@?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_5614732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_11_layer_call_fn_563910

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_5605432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
D__inference_dense_14_layer_call_and_return_conditional_losses_561383

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_8_layer_call_fn_564221

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_5609692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_563007
inputs_category_embed_1
inputs_category_embed_2
inputs_category_embed_3
inputs_category_embed_4
inputs_category_embed_5
inputs_city_embed_1
inputs_city_embed_2
inputs_city_embed_3
inputs_city_embed_4
inputs_city_embed_5
inputs_colon
inputs_commas
inputs_dash
inputs_exclam
inputs_money
inputs_month
inputs_parenthesis
inputs_state_embed_1
inputs_state_embed_2
inputs_state_embed_3
inputs_state_embed_4
inputs_state_embed_5
inputs_weekday9
'dense_14_matmul_readvariableop_resource: 6
(dense_14_biasadd_readvariableop_resource: A
3batch_normalization_11_cast_readvariableop_resource: C
5batch_normalization_11_cast_1_readvariableop_resource: C
5batch_normalization_11_cast_2_readvariableop_resource: C
5batch_normalization_11_cast_3_readvariableop_resource: :
'dense_13_matmul_readvariableop_resource:	 ?7
(dense_13_biasadd_readvariableop_resource:	?B
3batch_normalization_10_cast_readvariableop_resource:	?D
5batch_normalization_10_cast_1_readvariableop_resource:	?D
5batch_normalization_10_cast_2_readvariableop_resource:	?D
5batch_normalization_10_cast_3_readvariableop_resource:	?:
'dense_12_matmul_readvariableop_resource:	?@6
(dense_12_biasadd_readvariableop_resource:@@
2batch_normalization_9_cast_readvariableop_resource:@B
4batch_normalization_9_cast_1_readvariableop_resource:@B
4batch_normalization_9_cast_2_readvariableop_resource:@B
4batch_normalization_9_cast_3_readvariableop_resource:@:
'dense_11_matmul_readvariableop_resource:	@?7
(dense_11_biasadd_readvariableop_resource:	?A
2batch_normalization_8_cast_readvariableop_resource:	?C
4batch_normalization_8_cast_1_readvariableop_resource:	?C
4batch_normalization_8_cast_2_readvariableop_resource:	?C
4batch_normalization_8_cast_3_readvariableop_resource:	?:
'dense_10_matmul_readvariableop_resource:	?	6
(dense_10_biasadd_readvariableop_resource:	
identity??*batch_normalization_10/Cast/ReadVariableOp?,batch_normalization_10/Cast_1/ReadVariableOp?,batch_normalization_10/Cast_2/ReadVariableOp?,batch_normalization_10/Cast_3/ReadVariableOp?*batch_normalization_11/Cast/ReadVariableOp?,batch_normalization_11/Cast_1/ReadVariableOp?,batch_normalization_11/Cast_2/ReadVariableOp?,batch_normalization_11/Cast_3/ReadVariableOp?)batch_normalization_8/Cast/ReadVariableOp?+batch_normalization_8/Cast_1/ReadVariableOp?+batch_normalization_8/Cast_2/ReadVariableOp?+batch_normalization_8/Cast_3/ReadVariableOp?)batch_normalization_9/Cast/ReadVariableOp?+batch_normalization_9/Cast_1/ReadVariableOp?+batch_normalization_9/Cast_2/ReadVariableOp?+batch_normalization_9/Cast_3/ReadVariableOp?dense_10/BiasAdd/ReadVariableOp?dense_10/MatMul/ReadVariableOp?dense_11/BiasAdd/ReadVariableOp?dense_11/MatMul/ReadVariableOp?dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?dense_13/BiasAdd/ReadVariableOp?dense_13/MatMul/ReadVariableOp?dense_14/BiasAdd/ReadVariableOp?dense_14/MatMul/ReadVariableOp?>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp?>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp?
'dense_features_1/category_embed_1/ShapeShapeinputs_category_embed_1*
T0*
_output_shapes
:2)
'dense_features_1/category_embed_1/Shape?
5dense_features_1/category_embed_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5dense_features_1/category_embed_1/strided_slice/stack?
7dense_features_1/category_embed_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7dense_features_1/category_embed_1/strided_slice/stack_1?
7dense_features_1/category_embed_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7dense_features_1/category_embed_1/strided_slice/stack_2?
/dense_features_1/category_embed_1/strided_sliceStridedSlice0dense_features_1/category_embed_1/Shape:output:0>dense_features_1/category_embed_1/strided_slice/stack:output:0@dense_features_1/category_embed_1/strided_slice/stack_1:output:0@dense_features_1/category_embed_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/dense_features_1/category_embed_1/strided_slice?
1dense_features_1/category_embed_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :23
1dense_features_1/category_embed_1/Reshape/shape/1?
/dense_features_1/category_embed_1/Reshape/shapePack8dense_features_1/category_embed_1/strided_slice:output:0:dense_features_1/category_embed_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:21
/dense_features_1/category_embed_1/Reshape/shape?
)dense_features_1/category_embed_1/ReshapeReshapeinputs_category_embed_18dense_features_1/category_embed_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2+
)dense_features_1/category_embed_1/Reshape?
'dense_features_1/category_embed_2/ShapeShapeinputs_category_embed_2*
T0*
_output_shapes
:2)
'dense_features_1/category_embed_2/Shape?
5dense_features_1/category_embed_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5dense_features_1/category_embed_2/strided_slice/stack?
7dense_features_1/category_embed_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7dense_features_1/category_embed_2/strided_slice/stack_1?
7dense_features_1/category_embed_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7dense_features_1/category_embed_2/strided_slice/stack_2?
/dense_features_1/category_embed_2/strided_sliceStridedSlice0dense_features_1/category_embed_2/Shape:output:0>dense_features_1/category_embed_2/strided_slice/stack:output:0@dense_features_1/category_embed_2/strided_slice/stack_1:output:0@dense_features_1/category_embed_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/dense_features_1/category_embed_2/strided_slice?
1dense_features_1/category_embed_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :23
1dense_features_1/category_embed_2/Reshape/shape/1?
/dense_features_1/category_embed_2/Reshape/shapePack8dense_features_1/category_embed_2/strided_slice:output:0:dense_features_1/category_embed_2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:21
/dense_features_1/category_embed_2/Reshape/shape?
)dense_features_1/category_embed_2/ReshapeReshapeinputs_category_embed_28dense_features_1/category_embed_2/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2+
)dense_features_1/category_embed_2/Reshape?
'dense_features_1/category_embed_3/ShapeShapeinputs_category_embed_3*
T0*
_output_shapes
:2)
'dense_features_1/category_embed_3/Shape?
5dense_features_1/category_embed_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5dense_features_1/category_embed_3/strided_slice/stack?
7dense_features_1/category_embed_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7dense_features_1/category_embed_3/strided_slice/stack_1?
7dense_features_1/category_embed_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7dense_features_1/category_embed_3/strided_slice/stack_2?
/dense_features_1/category_embed_3/strided_sliceStridedSlice0dense_features_1/category_embed_3/Shape:output:0>dense_features_1/category_embed_3/strided_slice/stack:output:0@dense_features_1/category_embed_3/strided_slice/stack_1:output:0@dense_features_1/category_embed_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/dense_features_1/category_embed_3/strided_slice?
1dense_features_1/category_embed_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :23
1dense_features_1/category_embed_3/Reshape/shape/1?
/dense_features_1/category_embed_3/Reshape/shapePack8dense_features_1/category_embed_3/strided_slice:output:0:dense_features_1/category_embed_3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:21
/dense_features_1/category_embed_3/Reshape/shape?
)dense_features_1/category_embed_3/ReshapeReshapeinputs_category_embed_38dense_features_1/category_embed_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2+
)dense_features_1/category_embed_3/Reshape?
'dense_features_1/category_embed_4/ShapeShapeinputs_category_embed_4*
T0*
_output_shapes
:2)
'dense_features_1/category_embed_4/Shape?
5dense_features_1/category_embed_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5dense_features_1/category_embed_4/strided_slice/stack?
7dense_features_1/category_embed_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7dense_features_1/category_embed_4/strided_slice/stack_1?
7dense_features_1/category_embed_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7dense_features_1/category_embed_4/strided_slice/stack_2?
/dense_features_1/category_embed_4/strided_sliceStridedSlice0dense_features_1/category_embed_4/Shape:output:0>dense_features_1/category_embed_4/strided_slice/stack:output:0@dense_features_1/category_embed_4/strided_slice/stack_1:output:0@dense_features_1/category_embed_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/dense_features_1/category_embed_4/strided_slice?
1dense_features_1/category_embed_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :23
1dense_features_1/category_embed_4/Reshape/shape/1?
/dense_features_1/category_embed_4/Reshape/shapePack8dense_features_1/category_embed_4/strided_slice:output:0:dense_features_1/category_embed_4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:21
/dense_features_1/category_embed_4/Reshape/shape?
)dense_features_1/category_embed_4/ReshapeReshapeinputs_category_embed_48dense_features_1/category_embed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2+
)dense_features_1/category_embed_4/Reshape?
'dense_features_1/category_embed_5/ShapeShapeinputs_category_embed_5*
T0*
_output_shapes
:2)
'dense_features_1/category_embed_5/Shape?
5dense_features_1/category_embed_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5dense_features_1/category_embed_5/strided_slice/stack?
7dense_features_1/category_embed_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7dense_features_1/category_embed_5/strided_slice/stack_1?
7dense_features_1/category_embed_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7dense_features_1/category_embed_5/strided_slice/stack_2?
/dense_features_1/category_embed_5/strided_sliceStridedSlice0dense_features_1/category_embed_5/Shape:output:0>dense_features_1/category_embed_5/strided_slice/stack:output:0@dense_features_1/category_embed_5/strided_slice/stack_1:output:0@dense_features_1/category_embed_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/dense_features_1/category_embed_5/strided_slice?
1dense_features_1/category_embed_5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :23
1dense_features_1/category_embed_5/Reshape/shape/1?
/dense_features_1/category_embed_5/Reshape/shapePack8dense_features_1/category_embed_5/strided_slice:output:0:dense_features_1/category_embed_5/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:21
/dense_features_1/category_embed_5/Reshape/shape?
)dense_features_1/category_embed_5/ReshapeReshapeinputs_category_embed_58dense_features_1/category_embed_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2+
)dense_features_1/category_embed_5/Reshape?
#dense_features_1/city_embed_1/ShapeShapeinputs_city_embed_1*
T0*
_output_shapes
:2%
#dense_features_1/city_embed_1/Shape?
1dense_features_1/city_embed_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_features_1/city_embed_1/strided_slice/stack?
3dense_features_1/city_embed_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_features_1/city_embed_1/strided_slice/stack_1?
3dense_features_1/city_embed_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_features_1/city_embed_1/strided_slice/stack_2?
+dense_features_1/city_embed_1/strided_sliceStridedSlice,dense_features_1/city_embed_1/Shape:output:0:dense_features_1/city_embed_1/strided_slice/stack:output:0<dense_features_1/city_embed_1/strided_slice/stack_1:output:0<dense_features_1/city_embed_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_features_1/city_embed_1/strided_slice?
-dense_features_1/city_embed_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2/
-dense_features_1/city_embed_1/Reshape/shape/1?
+dense_features_1/city_embed_1/Reshape/shapePack4dense_features_1/city_embed_1/strided_slice:output:06dense_features_1/city_embed_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2-
+dense_features_1/city_embed_1/Reshape/shape?
%dense_features_1/city_embed_1/ReshapeReshapeinputs_city_embed_14dense_features_1/city_embed_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2'
%dense_features_1/city_embed_1/Reshape?
#dense_features_1/city_embed_2/ShapeShapeinputs_city_embed_2*
T0*
_output_shapes
:2%
#dense_features_1/city_embed_2/Shape?
1dense_features_1/city_embed_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_features_1/city_embed_2/strided_slice/stack?
3dense_features_1/city_embed_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_features_1/city_embed_2/strided_slice/stack_1?
3dense_features_1/city_embed_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_features_1/city_embed_2/strided_slice/stack_2?
+dense_features_1/city_embed_2/strided_sliceStridedSlice,dense_features_1/city_embed_2/Shape:output:0:dense_features_1/city_embed_2/strided_slice/stack:output:0<dense_features_1/city_embed_2/strided_slice/stack_1:output:0<dense_features_1/city_embed_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_features_1/city_embed_2/strided_slice?
-dense_features_1/city_embed_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2/
-dense_features_1/city_embed_2/Reshape/shape/1?
+dense_features_1/city_embed_2/Reshape/shapePack4dense_features_1/city_embed_2/strided_slice:output:06dense_features_1/city_embed_2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2-
+dense_features_1/city_embed_2/Reshape/shape?
%dense_features_1/city_embed_2/ReshapeReshapeinputs_city_embed_24dense_features_1/city_embed_2/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2'
%dense_features_1/city_embed_2/Reshape?
#dense_features_1/city_embed_3/ShapeShapeinputs_city_embed_3*
T0*
_output_shapes
:2%
#dense_features_1/city_embed_3/Shape?
1dense_features_1/city_embed_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_features_1/city_embed_3/strided_slice/stack?
3dense_features_1/city_embed_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_features_1/city_embed_3/strided_slice/stack_1?
3dense_features_1/city_embed_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_features_1/city_embed_3/strided_slice/stack_2?
+dense_features_1/city_embed_3/strided_sliceStridedSlice,dense_features_1/city_embed_3/Shape:output:0:dense_features_1/city_embed_3/strided_slice/stack:output:0<dense_features_1/city_embed_3/strided_slice/stack_1:output:0<dense_features_1/city_embed_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_features_1/city_embed_3/strided_slice?
-dense_features_1/city_embed_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2/
-dense_features_1/city_embed_3/Reshape/shape/1?
+dense_features_1/city_embed_3/Reshape/shapePack4dense_features_1/city_embed_3/strided_slice:output:06dense_features_1/city_embed_3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2-
+dense_features_1/city_embed_3/Reshape/shape?
%dense_features_1/city_embed_3/ReshapeReshapeinputs_city_embed_34dense_features_1/city_embed_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2'
%dense_features_1/city_embed_3/Reshape?
#dense_features_1/city_embed_4/ShapeShapeinputs_city_embed_4*
T0*
_output_shapes
:2%
#dense_features_1/city_embed_4/Shape?
1dense_features_1/city_embed_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_features_1/city_embed_4/strided_slice/stack?
3dense_features_1/city_embed_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_features_1/city_embed_4/strided_slice/stack_1?
3dense_features_1/city_embed_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_features_1/city_embed_4/strided_slice/stack_2?
+dense_features_1/city_embed_4/strided_sliceStridedSlice,dense_features_1/city_embed_4/Shape:output:0:dense_features_1/city_embed_4/strided_slice/stack:output:0<dense_features_1/city_embed_4/strided_slice/stack_1:output:0<dense_features_1/city_embed_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_features_1/city_embed_4/strided_slice?
-dense_features_1/city_embed_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2/
-dense_features_1/city_embed_4/Reshape/shape/1?
+dense_features_1/city_embed_4/Reshape/shapePack4dense_features_1/city_embed_4/strided_slice:output:06dense_features_1/city_embed_4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2-
+dense_features_1/city_embed_4/Reshape/shape?
%dense_features_1/city_embed_4/ReshapeReshapeinputs_city_embed_44dense_features_1/city_embed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2'
%dense_features_1/city_embed_4/Reshape?
#dense_features_1/city_embed_5/ShapeShapeinputs_city_embed_5*
T0*
_output_shapes
:2%
#dense_features_1/city_embed_5/Shape?
1dense_features_1/city_embed_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_features_1/city_embed_5/strided_slice/stack?
3dense_features_1/city_embed_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_features_1/city_embed_5/strided_slice/stack_1?
3dense_features_1/city_embed_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_features_1/city_embed_5/strided_slice/stack_2?
+dense_features_1/city_embed_5/strided_sliceStridedSlice,dense_features_1/city_embed_5/Shape:output:0:dense_features_1/city_embed_5/strided_slice/stack:output:0<dense_features_1/city_embed_5/strided_slice/stack_1:output:0<dense_features_1/city_embed_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_features_1/city_embed_5/strided_slice?
-dense_features_1/city_embed_5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2/
-dense_features_1/city_embed_5/Reshape/shape/1?
+dense_features_1/city_embed_5/Reshape/shapePack4dense_features_1/city_embed_5/strided_slice:output:06dense_features_1/city_embed_5/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2-
+dense_features_1/city_embed_5/Reshape/shape?
%dense_features_1/city_embed_5/ReshapeReshapeinputs_city_embed_54dense_features_1/city_embed_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2'
%dense_features_1/city_embed_5/Reshapex
dense_features_1/colon/ShapeShapeinputs_colon*
T0*
_output_shapes
:2
dense_features_1/colon/Shape?
*dense_features_1/colon/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*dense_features_1/colon/strided_slice/stack?
,dense_features_1/colon/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,dense_features_1/colon/strided_slice/stack_1?
,dense_features_1/colon/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,dense_features_1/colon/strided_slice/stack_2?
$dense_features_1/colon/strided_sliceStridedSlice%dense_features_1/colon/Shape:output:03dense_features_1/colon/strided_slice/stack:output:05dense_features_1/colon/strided_slice/stack_1:output:05dense_features_1/colon/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$dense_features_1/colon/strided_slice?
&dense_features_1/colon/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&dense_features_1/colon/Reshape/shape/1?
$dense_features_1/colon/Reshape/shapePack-dense_features_1/colon/strided_slice:output:0/dense_features_1/colon/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2&
$dense_features_1/colon/Reshape/shape?
dense_features_1/colon/ReshapeReshapeinputs_colon-dense_features_1/colon/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2 
dense_features_1/colon/Reshape{
dense_features_1/commas/ShapeShapeinputs_commas*
T0*
_output_shapes
:2
dense_features_1/commas/Shape?
+dense_features_1/commas/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+dense_features_1/commas/strided_slice/stack?
-dense_features_1/commas/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-dense_features_1/commas/strided_slice/stack_1?
-dense_features_1/commas/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-dense_features_1/commas/strided_slice/stack_2?
%dense_features_1/commas/strided_sliceStridedSlice&dense_features_1/commas/Shape:output:04dense_features_1/commas/strided_slice/stack:output:06dense_features_1/commas/strided_slice/stack_1:output:06dense_features_1/commas/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%dense_features_1/commas/strided_slice?
'dense_features_1/commas/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'dense_features_1/commas/Reshape/shape/1?
%dense_features_1/commas/Reshape/shapePack.dense_features_1/commas/strided_slice:output:00dense_features_1/commas/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2'
%dense_features_1/commas/Reshape/shape?
dense_features_1/commas/ReshapeReshapeinputs_commas.dense_features_1/commas/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2!
dense_features_1/commas/Reshapeu
dense_features_1/dash/ShapeShapeinputs_dash*
T0*
_output_shapes
:2
dense_features_1/dash/Shape?
)dense_features_1/dash/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)dense_features_1/dash/strided_slice/stack?
+dense_features_1/dash/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+dense_features_1/dash/strided_slice/stack_1?
+dense_features_1/dash/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+dense_features_1/dash/strided_slice/stack_2?
#dense_features_1/dash/strided_sliceStridedSlice$dense_features_1/dash/Shape:output:02dense_features_1/dash/strided_slice/stack:output:04dense_features_1/dash/strided_slice/stack_1:output:04dense_features_1/dash/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#dense_features_1/dash/strided_slice?
%dense_features_1/dash/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%dense_features_1/dash/Reshape/shape/1?
#dense_features_1/dash/Reshape/shapePack,dense_features_1/dash/strided_slice:output:0.dense_features_1/dash/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2%
#dense_features_1/dash/Reshape/shape?
dense_features_1/dash/ReshapeReshapeinputs_dash,dense_features_1/dash/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
dense_features_1/dash/Reshape{
dense_features_1/exclam/ShapeShapeinputs_exclam*
T0*
_output_shapes
:2
dense_features_1/exclam/Shape?
+dense_features_1/exclam/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+dense_features_1/exclam/strided_slice/stack?
-dense_features_1/exclam/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-dense_features_1/exclam/strided_slice/stack_1?
-dense_features_1/exclam/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-dense_features_1/exclam/strided_slice/stack_2?
%dense_features_1/exclam/strided_sliceStridedSlice&dense_features_1/exclam/Shape:output:04dense_features_1/exclam/strided_slice/stack:output:06dense_features_1/exclam/strided_slice/stack_1:output:06dense_features_1/exclam/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%dense_features_1/exclam/strided_slice?
'dense_features_1/exclam/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'dense_features_1/exclam/Reshape/shape/1?
%dense_features_1/exclam/Reshape/shapePack.dense_features_1/exclam/strided_slice:output:00dense_features_1/exclam/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2'
%dense_features_1/exclam/Reshape/shape?
dense_features_1/exclam/ReshapeReshapeinputs_exclam.dense_features_1/exclam/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2!
dense_features_1/exclam/Reshapex
dense_features_1/money/ShapeShapeinputs_money*
T0*
_output_shapes
:2
dense_features_1/money/Shape?
*dense_features_1/money/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*dense_features_1/money/strided_slice/stack?
,dense_features_1/money/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,dense_features_1/money/strided_slice/stack_1?
,dense_features_1/money/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,dense_features_1/money/strided_slice/stack_2?
$dense_features_1/money/strided_sliceStridedSlice%dense_features_1/money/Shape:output:03dense_features_1/money/strided_slice/stack:output:05dense_features_1/money/strided_slice/stack_1:output:05dense_features_1/money/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$dense_features_1/money/strided_slice?
&dense_features_1/money/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&dense_features_1/money/Reshape/shape/1?
$dense_features_1/money/Reshape/shapePack-dense_features_1/money/strided_slice:output:0/dense_features_1/money/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2&
$dense_features_1/money/Reshape/shape?
dense_features_1/money/ReshapeReshapeinputs_money-dense_features_1/money/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2 
dense_features_1/money/Reshapex
dense_features_1/month/ShapeShapeinputs_month*
T0*
_output_shapes
:2
dense_features_1/month/Shape?
*dense_features_1/month/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*dense_features_1/month/strided_slice/stack?
,dense_features_1/month/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,dense_features_1/month/strided_slice/stack_1?
,dense_features_1/month/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,dense_features_1/month/strided_slice/stack_2?
$dense_features_1/month/strided_sliceStridedSlice%dense_features_1/month/Shape:output:03dense_features_1/month/strided_slice/stack:output:05dense_features_1/month/strided_slice/stack_1:output:05dense_features_1/month/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$dense_features_1/month/strided_slice?
&dense_features_1/month/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&dense_features_1/month/Reshape/shape/1?
$dense_features_1/month/Reshape/shapePack-dense_features_1/month/strided_slice:output:0/dense_features_1/month/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2&
$dense_features_1/month/Reshape/shape?
dense_features_1/month/ReshapeReshapeinputs_month-dense_features_1/month/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2 
dense_features_1/month/Reshape?
"dense_features_1/parenthesis/ShapeShapeinputs_parenthesis*
T0*
_output_shapes
:2$
"dense_features_1/parenthesis/Shape?
0dense_features_1/parenthesis/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_features_1/parenthesis/strided_slice/stack?
2dense_features_1/parenthesis/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_features_1/parenthesis/strided_slice/stack_1?
2dense_features_1/parenthesis/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_features_1/parenthesis/strided_slice/stack_2?
*dense_features_1/parenthesis/strided_sliceStridedSlice+dense_features_1/parenthesis/Shape:output:09dense_features_1/parenthesis/strided_slice/stack:output:0;dense_features_1/parenthesis/strided_slice/stack_1:output:0;dense_features_1/parenthesis/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_features_1/parenthesis/strided_slice?
,dense_features_1/parenthesis/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2.
,dense_features_1/parenthesis/Reshape/shape/1?
*dense_features_1/parenthesis/Reshape/shapePack3dense_features_1/parenthesis/strided_slice:output:05dense_features_1/parenthesis/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2,
*dense_features_1/parenthesis/Reshape/shape?
$dense_features_1/parenthesis/ReshapeReshapeinputs_parenthesis3dense_features_1/parenthesis/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2&
$dense_features_1/parenthesis/Reshape?
$dense_features_1/state_embed_1/ShapeShapeinputs_state_embed_1*
T0*
_output_shapes
:2&
$dense_features_1/state_embed_1/Shape?
2dense_features_1/state_embed_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2dense_features_1/state_embed_1/strided_slice/stack?
4dense_features_1/state_embed_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4dense_features_1/state_embed_1/strided_slice/stack_1?
4dense_features_1/state_embed_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4dense_features_1/state_embed_1/strided_slice/stack_2?
,dense_features_1/state_embed_1/strided_sliceStridedSlice-dense_features_1/state_embed_1/Shape:output:0;dense_features_1/state_embed_1/strided_slice/stack:output:0=dense_features_1/state_embed_1/strided_slice/stack_1:output:0=dense_features_1/state_embed_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,dense_features_1/state_embed_1/strided_slice?
.dense_features_1/state_embed_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :20
.dense_features_1/state_embed_1/Reshape/shape/1?
,dense_features_1/state_embed_1/Reshape/shapePack5dense_features_1/state_embed_1/strided_slice:output:07dense_features_1/state_embed_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2.
,dense_features_1/state_embed_1/Reshape/shape?
&dense_features_1/state_embed_1/ReshapeReshapeinputs_state_embed_15dense_features_1/state_embed_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2(
&dense_features_1/state_embed_1/Reshape?
$dense_features_1/state_embed_2/ShapeShapeinputs_state_embed_2*
T0*
_output_shapes
:2&
$dense_features_1/state_embed_2/Shape?
2dense_features_1/state_embed_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2dense_features_1/state_embed_2/strided_slice/stack?
4dense_features_1/state_embed_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4dense_features_1/state_embed_2/strided_slice/stack_1?
4dense_features_1/state_embed_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4dense_features_1/state_embed_2/strided_slice/stack_2?
,dense_features_1/state_embed_2/strided_sliceStridedSlice-dense_features_1/state_embed_2/Shape:output:0;dense_features_1/state_embed_2/strided_slice/stack:output:0=dense_features_1/state_embed_2/strided_slice/stack_1:output:0=dense_features_1/state_embed_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,dense_features_1/state_embed_2/strided_slice?
.dense_features_1/state_embed_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :20
.dense_features_1/state_embed_2/Reshape/shape/1?
,dense_features_1/state_embed_2/Reshape/shapePack5dense_features_1/state_embed_2/strided_slice:output:07dense_features_1/state_embed_2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2.
,dense_features_1/state_embed_2/Reshape/shape?
&dense_features_1/state_embed_2/ReshapeReshapeinputs_state_embed_25dense_features_1/state_embed_2/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2(
&dense_features_1/state_embed_2/Reshape?
$dense_features_1/state_embed_3/ShapeShapeinputs_state_embed_3*
T0*
_output_shapes
:2&
$dense_features_1/state_embed_3/Shape?
2dense_features_1/state_embed_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2dense_features_1/state_embed_3/strided_slice/stack?
4dense_features_1/state_embed_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4dense_features_1/state_embed_3/strided_slice/stack_1?
4dense_features_1/state_embed_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4dense_features_1/state_embed_3/strided_slice/stack_2?
,dense_features_1/state_embed_3/strided_sliceStridedSlice-dense_features_1/state_embed_3/Shape:output:0;dense_features_1/state_embed_3/strided_slice/stack:output:0=dense_features_1/state_embed_3/strided_slice/stack_1:output:0=dense_features_1/state_embed_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,dense_features_1/state_embed_3/strided_slice?
.dense_features_1/state_embed_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :20
.dense_features_1/state_embed_3/Reshape/shape/1?
,dense_features_1/state_embed_3/Reshape/shapePack5dense_features_1/state_embed_3/strided_slice:output:07dense_features_1/state_embed_3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2.
,dense_features_1/state_embed_3/Reshape/shape?
&dense_features_1/state_embed_3/ReshapeReshapeinputs_state_embed_35dense_features_1/state_embed_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2(
&dense_features_1/state_embed_3/Reshape?
$dense_features_1/state_embed_4/ShapeShapeinputs_state_embed_4*
T0*
_output_shapes
:2&
$dense_features_1/state_embed_4/Shape?
2dense_features_1/state_embed_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2dense_features_1/state_embed_4/strided_slice/stack?
4dense_features_1/state_embed_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4dense_features_1/state_embed_4/strided_slice/stack_1?
4dense_features_1/state_embed_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4dense_features_1/state_embed_4/strided_slice/stack_2?
,dense_features_1/state_embed_4/strided_sliceStridedSlice-dense_features_1/state_embed_4/Shape:output:0;dense_features_1/state_embed_4/strided_slice/stack:output:0=dense_features_1/state_embed_4/strided_slice/stack_1:output:0=dense_features_1/state_embed_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,dense_features_1/state_embed_4/strided_slice?
.dense_features_1/state_embed_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :20
.dense_features_1/state_embed_4/Reshape/shape/1?
,dense_features_1/state_embed_4/Reshape/shapePack5dense_features_1/state_embed_4/strided_slice:output:07dense_features_1/state_embed_4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2.
,dense_features_1/state_embed_4/Reshape/shape?
&dense_features_1/state_embed_4/ReshapeReshapeinputs_state_embed_45dense_features_1/state_embed_4/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2(
&dense_features_1/state_embed_4/Reshape?
$dense_features_1/state_embed_5/ShapeShapeinputs_state_embed_5*
T0*
_output_shapes
:2&
$dense_features_1/state_embed_5/Shape?
2dense_features_1/state_embed_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2dense_features_1/state_embed_5/strided_slice/stack?
4dense_features_1/state_embed_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4dense_features_1/state_embed_5/strided_slice/stack_1?
4dense_features_1/state_embed_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4dense_features_1/state_embed_5/strided_slice/stack_2?
,dense_features_1/state_embed_5/strided_sliceStridedSlice-dense_features_1/state_embed_5/Shape:output:0;dense_features_1/state_embed_5/strided_slice/stack:output:0=dense_features_1/state_embed_5/strided_slice/stack_1:output:0=dense_features_1/state_embed_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,dense_features_1/state_embed_5/strided_slice?
.dense_features_1/state_embed_5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :20
.dense_features_1/state_embed_5/Reshape/shape/1?
,dense_features_1/state_embed_5/Reshape/shapePack5dense_features_1/state_embed_5/strided_slice:output:07dense_features_1/state_embed_5/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2.
,dense_features_1/state_embed_5/Reshape/shape?
&dense_features_1/state_embed_5/ReshapeReshapeinputs_state_embed_55dense_features_1/state_embed_5/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2(
&dense_features_1/state_embed_5/Reshape~
dense_features_1/weekday/ShapeShapeinputs_weekday*
T0*
_output_shapes
:2 
dense_features_1/weekday/Shape?
,dense_features_1/weekday/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,dense_features_1/weekday/strided_slice/stack?
.dense_features_1/weekday/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.dense_features_1/weekday/strided_slice/stack_1?
.dense_features_1/weekday/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.dense_features_1/weekday/strided_slice/stack_2?
&dense_features_1/weekday/strided_sliceStridedSlice'dense_features_1/weekday/Shape:output:05dense_features_1/weekday/strided_slice/stack:output:07dense_features_1/weekday/strided_slice/stack_1:output:07dense_features_1/weekday/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&dense_features_1/weekday/strided_slice?
(dense_features_1/weekday/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(dense_features_1/weekday/Reshape/shape/1?
&dense_features_1/weekday/Reshape/shapePack/dense_features_1/weekday/strided_slice:output:01dense_features_1/weekday/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2(
&dense_features_1/weekday/Reshape/shape?
 dense_features_1/weekday/ReshapeReshapeinputs_weekday/dense_features_1/weekday/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2"
 dense_features_1/weekday/Reshape?
dense_features_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
dense_features_1/concat/axis?	
dense_features_1/concatConcatV22dense_features_1/category_embed_1/Reshape:output:02dense_features_1/category_embed_2/Reshape:output:02dense_features_1/category_embed_3/Reshape:output:02dense_features_1/category_embed_4/Reshape:output:02dense_features_1/category_embed_5/Reshape:output:0.dense_features_1/city_embed_1/Reshape:output:0.dense_features_1/city_embed_2/Reshape:output:0.dense_features_1/city_embed_3/Reshape:output:0.dense_features_1/city_embed_4/Reshape:output:0.dense_features_1/city_embed_5/Reshape:output:0'dense_features_1/colon/Reshape:output:0(dense_features_1/commas/Reshape:output:0&dense_features_1/dash/Reshape:output:0(dense_features_1/exclam/Reshape:output:0'dense_features_1/money/Reshape:output:0'dense_features_1/month/Reshape:output:0-dense_features_1/parenthesis/Reshape:output:0/dense_features_1/state_embed_1/Reshape:output:0/dense_features_1/state_embed_2/Reshape:output:0/dense_features_1/state_embed_3/Reshape:output:0/dense_features_1/state_embed_4/Reshape:output:0/dense_features_1/state_embed_5/Reshape:output:0)dense_features_1/weekday/Reshape:output:0%dense_features_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
dense_features_1/concat?
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_14/MatMul/ReadVariableOp?
dense_14/MatMulMatMul dense_features_1/concat:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_14/MatMul?
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_14/BiasAdd/ReadVariableOp?
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_14/BiasAdds
dense_14/ReluReludense_14/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_14/Relu?
*batch_normalization_11/Cast/ReadVariableOpReadVariableOp3batch_normalization_11_cast_readvariableop_resource*
_output_shapes
: *
dtype02,
*batch_normalization_11/Cast/ReadVariableOp?
,batch_normalization_11/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_11_cast_1_readvariableop_resource*
_output_shapes
: *
dtype02.
,batch_normalization_11/Cast_1/ReadVariableOp?
,batch_normalization_11/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_11_cast_2_readvariableop_resource*
_output_shapes
: *
dtype02.
,batch_normalization_11/Cast_2/ReadVariableOp?
,batch_normalization_11/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_11_cast_3_readvariableop_resource*
_output_shapes
: *
dtype02.
,batch_normalization_11/Cast_3/ReadVariableOp?
&batch_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_11/batchnorm/add/y?
$batch_normalization_11/batchnorm/addAddV24batch_normalization_11/Cast_1/ReadVariableOp:value:0/batch_normalization_11/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2&
$batch_normalization_11/batchnorm/add?
&batch_normalization_11/batchnorm/RsqrtRsqrt(batch_normalization_11/batchnorm/add:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_11/batchnorm/Rsqrt?
$batch_normalization_11/batchnorm/mulMul*batch_normalization_11/batchnorm/Rsqrt:y:04batch_normalization_11/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
: 2&
$batch_normalization_11/batchnorm/mul?
&batch_normalization_11/batchnorm/mul_1Muldense_14/Relu:activations:0(batch_normalization_11/batchnorm/mul:z:0*
T0*'
_output_shapes
:????????? 2(
&batch_normalization_11/batchnorm/mul_1?
&batch_normalization_11/batchnorm/mul_2Mul2batch_normalization_11/Cast/ReadVariableOp:value:0(batch_normalization_11/batchnorm/mul:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_11/batchnorm/mul_2?
$batch_normalization_11/batchnorm/subSub4batch_normalization_11/Cast_2/ReadVariableOp:value:0*batch_normalization_11/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2&
$batch_normalization_11/batchnorm/sub?
&batch_normalization_11/batchnorm/add_1AddV2*batch_normalization_11/batchnorm/mul_1:z:0(batch_normalization_11/batchnorm/sub:z:0*
T0*'
_output_shapes
:????????? 2(
&batch_normalization_11/batchnorm/add_1?
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02 
dense_13/MatMul/ReadVariableOp?
dense_13/MatMulMatMul*batch_normalization_11/batchnorm/add_1:z:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_13/MatMul?
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_13/BiasAdd/ReadVariableOp?
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_13/BiasAddt
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_13/Relu?
*batch_normalization_10/Cast/ReadVariableOpReadVariableOp3batch_normalization_10_cast_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*batch_normalization_10/Cast/ReadVariableOp?
,batch_normalization_10/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_10_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,batch_normalization_10/Cast_1/ReadVariableOp?
,batch_normalization_10/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_10_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,batch_normalization_10/Cast_2/ReadVariableOp?
,batch_normalization_10/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_10_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,batch_normalization_10/Cast_3/ReadVariableOp?
&batch_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_10/batchnorm/add/y?
$batch_normalization_10/batchnorm/addAddV24batch_normalization_10/Cast_1/ReadVariableOp:value:0/batch_normalization_10/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2&
$batch_normalization_10/batchnorm/add?
&batch_normalization_10/batchnorm/RsqrtRsqrt(batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_10/batchnorm/Rsqrt?
$batch_normalization_10/batchnorm/mulMul*batch_normalization_10/batchnorm/Rsqrt:y:04batch_normalization_10/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2&
$batch_normalization_10/batchnorm/mul?
&batch_normalization_10/batchnorm/mul_1Muldense_13/Relu:activations:0(batch_normalization_10/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2(
&batch_normalization_10/batchnorm/mul_1?
&batch_normalization_10/batchnorm/mul_2Mul2batch_normalization_10/Cast/ReadVariableOp:value:0(batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_10/batchnorm/mul_2?
$batch_normalization_10/batchnorm/subSub4batch_normalization_10/Cast_2/ReadVariableOp:value:0*batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2&
$batch_normalization_10/batchnorm/sub?
&batch_normalization_10/batchnorm/add_1AddV2*batch_normalization_10/batchnorm/mul_1:z:0(batch_normalization_10/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2(
&batch_normalization_10/batchnorm/add_1?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02 
dense_12/MatMul/ReadVariableOp?
dense_12/MatMulMatMul*batch_normalization_10/batchnorm/add_1:z:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_12/MatMul?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_12/BiasAdd/ReadVariableOp?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_12/BiasAdds
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_12/Relu?
)batch_normalization_9/Cast/ReadVariableOpReadVariableOp2batch_normalization_9_cast_readvariableop_resource*
_output_shapes
:@*
dtype02+
)batch_normalization_9/Cast/ReadVariableOp?
+batch_normalization_9/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_9_cast_1_readvariableop_resource*
_output_shapes
:@*
dtype02-
+batch_normalization_9/Cast_1/ReadVariableOp?
+batch_normalization_9/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_9_cast_2_readvariableop_resource*
_output_shapes
:@*
dtype02-
+batch_normalization_9/Cast_2/ReadVariableOp?
+batch_normalization_9/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_9_cast_3_readvariableop_resource*
_output_shapes
:@*
dtype02-
+batch_normalization_9/Cast_3/ReadVariableOp?
%batch_normalization_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%batch_normalization_9/batchnorm/add/y?
#batch_normalization_9/batchnorm/addAddV23batch_normalization_9/Cast_1/ReadVariableOp:value:0.batch_normalization_9/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_9/batchnorm/add?
%batch_normalization_9/batchnorm/RsqrtRsqrt'batch_normalization_9/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_9/batchnorm/Rsqrt?
#batch_normalization_9/batchnorm/mulMul)batch_normalization_9/batchnorm/Rsqrt:y:03batch_normalization_9/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_9/batchnorm/mul?
%batch_normalization_9/batchnorm/mul_1Muldense_12/Relu:activations:0'batch_normalization_9/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????@2'
%batch_normalization_9/batchnorm/mul_1?
%batch_normalization_9/batchnorm/mul_2Mul1batch_normalization_9/Cast/ReadVariableOp:value:0'batch_normalization_9/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_9/batchnorm/mul_2?
#batch_normalization_9/batchnorm/subSub3batch_normalization_9/Cast_2/ReadVariableOp:value:0)batch_normalization_9/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_9/batchnorm/sub?
%batch_normalization_9/batchnorm/add_1AddV2)batch_normalization_9/batchnorm/mul_1:z:0'batch_normalization_9/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????@2'
%batch_normalization_9/batchnorm/add_1?
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	@?*
dtype02 
dense_11/MatMul/ReadVariableOp?
dense_11/MatMulMatMul)batch_normalization_9/batchnorm/add_1:z:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_11/MatMul?
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_11/BiasAdd/ReadVariableOp?
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_11/BiasAddt
dense_11/ReluReludense_11/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_11/Relu?
)batch_normalization_8/Cast/ReadVariableOpReadVariableOp2batch_normalization_8_cast_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)batch_normalization_8/Cast/ReadVariableOp?
+batch_normalization_8/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_8_cast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+batch_normalization_8/Cast_1/ReadVariableOp?
+batch_normalization_8/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_8_cast_2_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+batch_normalization_8/Cast_2/ReadVariableOp?
+batch_normalization_8/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_8_cast_3_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+batch_normalization_8/Cast_3/ReadVariableOp?
%batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%batch_normalization_8/batchnorm/add/y?
#batch_normalization_8/batchnorm/addAddV23batch_normalization_8/Cast_1/ReadVariableOp:value:0.batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2%
#batch_normalization_8/batchnorm/add?
%batch_normalization_8/batchnorm/RsqrtRsqrt'batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes	
:?2'
%batch_normalization_8/batchnorm/Rsqrt?
#batch_normalization_8/batchnorm/mulMul)batch_normalization_8/batchnorm/Rsqrt:y:03batch_normalization_8/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2%
#batch_normalization_8/batchnorm/mul?
%batch_normalization_8/batchnorm/mul_1Muldense_11/Relu:activations:0'batch_normalization_8/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2'
%batch_normalization_8/batchnorm/mul_1?
%batch_normalization_8/batchnorm/mul_2Mul1batch_normalization_8/Cast/ReadVariableOp:value:0'batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2'
%batch_normalization_8/batchnorm/mul_2?
#batch_normalization_8/batchnorm/subSub3batch_normalization_8/Cast_2/ReadVariableOp:value:0)batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2%
#batch_normalization_8/batchnorm/sub?
%batch_normalization_8/batchnorm/add_1AddV2)batch_normalization_8/batchnorm/mul_1:z:0'batch_normalization_8/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2'
%batch_normalization_8/batchnorm/add_1?
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes
:	?	*
dtype02 
dense_10/MatMul/ReadVariableOp?
dense_10/MatMulMatMul)batch_normalization_8/batchnorm/add_1:z:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
dense_10/MatMul?
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02!
dense_10/BiasAdd/ReadVariableOp?
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
dense_10/BiasAdd|
dense_10/SoftmaxSoftmaxdense_10/BiasAdd:output:0*
T0*'
_output_shapes
:?????????	2
dense_10/Softmax?
>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02@
>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp?
/sequential_1/dense_13/kernel/Regularizer/SquareSquareFsequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 ?21
/sequential_1/dense_13/kernel/Regularizer/Square?
.sequential_1/dense_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       20
.sequential_1/dense_13/kernel/Regularizer/Const?
,sequential_1/dense_13/kernel/Regularizer/SumSum3sequential_1/dense_13/kernel/Regularizer/Square:y:07sequential_1/dense_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2.
,sequential_1/dense_13/kernel/Regularizer/Sum?
.sequential_1/dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.sequential_1/dense_13/kernel/Regularizer/mul/x?
,sequential_1/dense_13/kernel/Regularizer/mulMul7sequential_1/dense_13/kernel/Regularizer/mul/x:output:05sequential_1/dense_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,sequential_1/dense_13/kernel/Regularizer/mul?
>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02@
>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp?
/sequential_1/dense_12/kernel/Regularizer/SquareSquareFsequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?@21
/sequential_1/dense_12/kernel/Regularizer/Square?
.sequential_1/dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       20
.sequential_1/dense_12/kernel/Regularizer/Const?
,sequential_1/dense_12/kernel/Regularizer/SumSum3sequential_1/dense_12/kernel/Regularizer/Square:y:07sequential_1/dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2.
,sequential_1/dense_12/kernel/Regularizer/Sum?
.sequential_1/dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>20
.sequential_1/dense_12/kernel/Regularizer/mul/x?
,sequential_1/dense_12/kernel/Regularizer/mulMul7sequential_1/dense_12/kernel/Regularizer/mul/x:output:05sequential_1/dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,sequential_1/dense_12/kernel/Regularizer/mul?

IdentityIdentitydense_10/Softmax:softmax:0+^batch_normalization_10/Cast/ReadVariableOp-^batch_normalization_10/Cast_1/ReadVariableOp-^batch_normalization_10/Cast_2/ReadVariableOp-^batch_normalization_10/Cast_3/ReadVariableOp+^batch_normalization_11/Cast/ReadVariableOp-^batch_normalization_11/Cast_1/ReadVariableOp-^batch_normalization_11/Cast_2/ReadVariableOp-^batch_normalization_11/Cast_3/ReadVariableOp*^batch_normalization_8/Cast/ReadVariableOp,^batch_normalization_8/Cast_1/ReadVariableOp,^batch_normalization_8/Cast_2/ReadVariableOp,^batch_normalization_8/Cast_3/ReadVariableOp*^batch_normalization_9/Cast/ReadVariableOp,^batch_normalization_9/Cast_1/ReadVariableOp,^batch_normalization_9/Cast_2/ReadVariableOp,^batch_normalization_9/Cast_3/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp?^sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp?^sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2X
*batch_normalization_10/Cast/ReadVariableOp*batch_normalization_10/Cast/ReadVariableOp2\
,batch_normalization_10/Cast_1/ReadVariableOp,batch_normalization_10/Cast_1/ReadVariableOp2\
,batch_normalization_10/Cast_2/ReadVariableOp,batch_normalization_10/Cast_2/ReadVariableOp2\
,batch_normalization_10/Cast_3/ReadVariableOp,batch_normalization_10/Cast_3/ReadVariableOp2X
*batch_normalization_11/Cast/ReadVariableOp*batch_normalization_11/Cast/ReadVariableOp2\
,batch_normalization_11/Cast_1/ReadVariableOp,batch_normalization_11/Cast_1/ReadVariableOp2\
,batch_normalization_11/Cast_2/ReadVariableOp,batch_normalization_11/Cast_2/ReadVariableOp2\
,batch_normalization_11/Cast_3/ReadVariableOp,batch_normalization_11/Cast_3/ReadVariableOp2V
)batch_normalization_8/Cast/ReadVariableOp)batch_normalization_8/Cast/ReadVariableOp2Z
+batch_normalization_8/Cast_1/ReadVariableOp+batch_normalization_8/Cast_1/ReadVariableOp2Z
+batch_normalization_8/Cast_2/ReadVariableOp+batch_normalization_8/Cast_2/ReadVariableOp2Z
+batch_normalization_8/Cast_3/ReadVariableOp+batch_normalization_8/Cast_3/ReadVariableOp2V
)batch_normalization_9/Cast/ReadVariableOp)batch_normalization_9/Cast/ReadVariableOp2Z
+batch_normalization_9/Cast_1/ReadVariableOp+batch_normalization_9/Cast_1/ReadVariableOp2Z
+batch_normalization_9/Cast_2/ReadVariableOp+batch_normalization_9/Cast_2/ReadVariableOp2Z
+batch_normalization_9/Cast_3/ReadVariableOp+batch_normalization_9/Cast_3/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2?
>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp2?
>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp>sequential_1/dense_13/kernel/Regularizer/Square/ReadVariableOp:` \
'
_output_shapes
:?????????
1
_user_specified_nameinputs/category_embed_1:`\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/category_embed_2:`\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/category_embed_3:`\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/category_embed_4:`\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/category_embed_5:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/city_embed_1:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/city_embed_2:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/city_embed_3:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/city_embed_4:\	X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/city_embed_5:U
Q
'
_output_shapes
:?????????
&
_user_specified_nameinputs/colon:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/commas:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/dash:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/exclam:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/money:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/month:[W
'
_output_shapes
:?????????
,
_user_specified_nameinputs/parenthesis:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/state_embed_1:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/state_embed_2:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/state_embed_3:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/state_embed_4:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/state_embed_5:WS
'
_output_shapes
:?????????
(
_user_specified_nameinputs/weekday
?&
?	
$__inference_signature_wrapper_562526
category_embed_1
category_embed_2
category_embed_3
category_embed_4
category_embed_5
city_embed_1
city_embed_2
city_embed_3
city_embed_4
city_embed_5	
colon

commas
dash

exclam	
money	
month
parenthesis
state_embed_1
state_embed_2
state_embed_3
state_embed_4
state_embed_5
weekday
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:	 ?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?

unknown_11:	?@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:	@?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?

unknown_23:	?	

unknown_24:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallcategory_embed_1category_embed_2category_embed_3category_embed_4category_embed_5city_embed_1city_embed_2city_embed_3city_embed_4city_embed_5coloncommasdashexclammoneymonthparenthesisstate_embed_1state_embed_2state_embed_3state_embed_4state_embed_5weekdayunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*<
_read_only_resource_inputs
 !"#$%&'()*+,-./0*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_5604592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
'
_output_shapes
:?????????
*
_user_specified_namecategory_embed_1:YU
'
_output_shapes
:?????????
*
_user_specified_namecategory_embed_2:YU
'
_output_shapes
:?????????
*
_user_specified_namecategory_embed_3:YU
'
_output_shapes
:?????????
*
_user_specified_namecategory_embed_4:YU
'
_output_shapes
:?????????
*
_user_specified_namecategory_embed_5:UQ
'
_output_shapes
:?????????
&
_user_specified_namecity_embed_1:UQ
'
_output_shapes
:?????????
&
_user_specified_namecity_embed_2:UQ
'
_output_shapes
:?????????
&
_user_specified_namecity_embed_3:UQ
'
_output_shapes
:?????????
&
_user_specified_namecity_embed_4:U	Q
'
_output_shapes
:?????????
&
_user_specified_namecity_embed_5:N
J
'
_output_shapes
:?????????

_user_specified_namecolon:OK
'
_output_shapes
:?????????
 
_user_specified_namecommas:MI
'
_output_shapes
:?????????

_user_specified_namedash:OK
'
_output_shapes
:?????????
 
_user_specified_nameexclam:NJ
'
_output_shapes
:?????????

_user_specified_namemoney:NJ
'
_output_shapes
:?????????

_user_specified_namemonth:TP
'
_output_shapes
:?????????
%
_user_specified_nameparenthesis:VR
'
_output_shapes
:?????????
'
_user_specified_namestate_embed_1:VR
'
_output_shapes
:?????????
'
_user_specified_namestate_embed_2:VR
'
_output_shapes
:?????????
'
_user_specified_namestate_embed_3:VR
'
_output_shapes
:?????????
'
_user_specified_namestate_embed_4:VR
'
_output_shapes
:?????????
'
_user_specified_namestate_embed_5:PL
'
_output_shapes
:?????????
!
_user_specified_name	weekday
?
?
D__inference_dense_12_layer_call_and_return_conditional_losses_564108

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02@
>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp?
/sequential_1/dense_12/kernel/Regularizer/SquareSquareFsequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?@21
/sequential_1/dense_12/kernel/Regularizer/Square?
.sequential_1/dense_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       20
.sequential_1/dense_12/kernel/Regularizer/Const?
,sequential_1/dense_12/kernel/Regularizer/SumSum3sequential_1/dense_12/kernel/Regularizer/Square:y:07sequential_1/dense_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2.
,sequential_1/dense_12/kernel/Regularizer/Sum?
.sequential_1/dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>20
.sequential_1/dense_12/kernel/Regularizer/mul/x?
,sequential_1/dense_12/kernel/Regularizer/mulMul7sequential_1/dense_12/kernel/Regularizer/mul/x:output:05sequential_1/dense_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,sequential_1/dense_12/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp?^sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2?
>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp>sequential_1/dense_12/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
)__inference_dense_10_layer_call_fn_564297

inputs
unknown:	?	
	unknown_0:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_5614992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_564254

inputs+
cast_readvariableop_resource:	?-
cast_1_readvariableop_resource:	?-
cast_2_readvariableop_resource:	?-
cast_3_readvariableop_resource:	?
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:?*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
M
category_embed_19
"serving_default_category_embed_1:0?????????
M
category_embed_29
"serving_default_category_embed_2:0?????????
M
category_embed_39
"serving_default_category_embed_3:0?????????
M
category_embed_49
"serving_default_category_embed_4:0?????????
M
category_embed_59
"serving_default_category_embed_5:0?????????
E
city_embed_15
serving_default_city_embed_1:0?????????
E
city_embed_25
serving_default_city_embed_2:0?????????
E
city_embed_35
serving_default_city_embed_3:0?????????
E
city_embed_45
serving_default_city_embed_4:0?????????
E
city_embed_55
serving_default_city_embed_5:0?????????
7
colon.
serving_default_colon:0?????????
9
commas/
serving_default_commas:0?????????
5
dash-
serving_default_dash:0?????????
9
exclam/
serving_default_exclam:0?????????
7
money.
serving_default_money:0?????????
7
month.
serving_default_month:0?????????
C
parenthesis4
serving_default_parenthesis:0?????????
G
state_embed_16
serving_default_state_embed_1:0?????????
G
state_embed_26
serving_default_state_embed_2:0?????????
G
state_embed_36
serving_default_state_embed_3:0?????????
G
state_embed_46
serving_default_state_embed_4:0?????????
G
state_embed_56
serving_default_state_embed_5:0?????????
;
weekday0
serving_default_weekday:0?????????<
output_10
StatefulPartitionedCall:0?????????	tensorflow/serving/predict:??
??
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
	optimizer
_build_input_shape
	variables
regularization_losses
trainable_variables
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"??
_tf_keras_sequential??{"name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "DenseFeatures", "config": {"name": "dense_features_1", "trainable": true, "dtype": "float32", "feature_columns": [{"class_name": "NumericColumn", "config": {"key": "category_embed_1", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "category_embed_2", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "category_embed_3", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "category_embed_4", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "category_embed_5", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "city_embed_1", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "city_embed_2", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "city_embed_3", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "city_embed_4", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "city_embed_5", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "colon", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "commas", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "dash", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "exclam", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "money", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "month", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "parenthesis", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "state_embed_1", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "state_embed_2", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "state_embed_3", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "state_embed_4", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "state_embed_5", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "weekday", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}], "partitioner": null}}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.5}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.25}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 9, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": {"weekday": {"class_name": "__tuple__", "items": [null, 1]}, "month": {"class_name": "__tuple__", "items": [null, 1]}, "commas": {"class_name": "__tuple__", "items": [null, 1]}, "exclam": {"class_name": "__tuple__", "items": [null, 1]}, "money": {"class_name": "__tuple__", "items": [null, 1]}, "dash": {"class_name": "__tuple__", "items": [null, 1]}, "colon": {"class_name": "__tuple__", "items": [null, 1]}, "parenthesis": {"class_name": "__tuple__", "items": [null, 1]}, "city_embed_1": {"class_name": "__tuple__", "items": [null, 1]}, "city_embed_2": {"class_name": "__tuple__", "items": [null, 1]}, "city_embed_3": {"class_name": "__tuple__", "items": [null, 1]}, "city_embed_4": {"class_name": "__tuple__", "items": [null, 1]}, "city_embed_5": {"class_name": "__tuple__", "items": [null, 1]}, "state_embed_1": {"class_name": "__tuple__", "items": [null, 1]}, "state_embed_2": {"class_name": "__tuple__", "items": [null, 1]}, "state_embed_3": {"class_name": "__tuple__", "items": [null, 1]}, "state_embed_4": {"class_name": "__tuple__", "items": [null, 1]}, "state_embed_5": {"class_name": "__tuple__", "items": [null, 1]}, "category_embed_1": {"class_name": "__tuple__", "items": [null, 1]}, "category_embed_2": {"class_name": "__tuple__", "items": [null, 1]}, "category_embed_3": {"class_name": "__tuple__", "items": [null, 1]}, "category_embed_4": {"class_name": "__tuple__", "items": [null, 1]}, "category_embed_5": {"class_name": "__tuple__", "items": [null, 1]}}}, "shared_object_id": 38, "build_input_shape": {"weekday": {"class_name": "__tuple__", "items": [null, 1]}, "month": {"class_name": "__tuple__", "items": [null, 1]}, "commas": {"class_name": "__tuple__", "items": [null, 1]}, "exclam": {"class_name": "__tuple__", "items": [null, 1]}, "money": {"class_name": "__tuple__", "items": [null, 1]}, "dash": {"class_name": "__tuple__", "items": [null, 1]}, "colon": {"class_name": "__tuple__", "items": [null, 1]}, "parenthesis": {"class_name": "__tuple__", "items": [null, 1]}, "city_embed_1": {"class_name": "__tuple__", "items": [null, 1]}, "city_embed_2": {"class_name": "__tuple__", "items": [null, 1]}, "city_embed_3": {"class_name": "__tuple__", "items": [null, 1]}, "city_embed_4": {"class_name": "__tuple__", "items": [null, 1]}, "city_embed_5": {"class_name": "__tuple__", "items": [null, 1]}, "state_embed_1": {"class_name": "__tuple__", "items": [null, 1]}, "state_embed_2": {"class_name": "__tuple__", "items": [null, 1]}, "state_embed_3": {"class_name": "__tuple__", "items": [null, 1]}, "state_embed_4": {"class_name": "__tuple__", "items": [null, 1]}, "state_embed_5": {"class_name": "__tuple__", "items": [null, 1]}, "category_embed_1": {"class_name": "__tuple__", "items": [null, 1]}, "category_embed_2": {"class_name": "__tuple__", "items": [null, 1]}, "category_embed_3": {"class_name": "__tuple__", "items": [null, 1]}, "category_embed_4": {"class_name": "__tuple__", "items": [null, 1]}, "category_embed_5": {"class_name": "__tuple__", "items": [null, 1]}}, "is_graph_network": false, "save_spec": {"weekday": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "weekday"]}, "month": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "month"]}, "commas": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "commas"]}, "exclam": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "exclam"]}, "money": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "money"]}, "dash": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "dash"]}, "colon": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "colon"]}, "parenthesis": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "parenthesis"]}, "city_embed_1": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "city_embed_1"]}, "city_embed_2": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "city_embed_2"]}, "city_embed_3": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "city_embed_3"]}, "city_embed_4": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "city_embed_4"]}, "city_embed_5": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "city_embed_5"]}, "state_embed_1": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "state_embed_1"]}, "state_embed_2": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "state_embed_2"]}, "state_embed_3": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "state_embed_3"]}, "state_embed_4": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "state_embed_4"]}, "state_embed_5": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "state_embed_5"]}, "category_embed_1": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "category_embed_1"]}, "category_embed_2": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "category_embed_2"]}, "category_embed_3": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "category_embed_3"]}, "category_embed_4": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "category_embed_4"]}, "category_embed_5": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "category_embed_5"]}}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "DenseFeatures", "config": {"name": "dense_features_1", "trainable": true, "dtype": "float32", "feature_columns": [{"class_name": "NumericColumn", "config": {"key": "category_embed_1", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "category_embed_2", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "category_embed_3", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "category_embed_4", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "category_embed_5", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "city_embed_1", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "city_embed_2", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "city_embed_3", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "city_embed_4", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "city_embed_5", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "colon", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "commas", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "dash", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "exclam", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "money", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "month", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "parenthesis", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "state_embed_1", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "state_embed_2", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "state_embed_3", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "state_embed_4", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "state_embed_5", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "weekday", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}], "partitioner": null}, "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.5}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 14}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 16}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 17}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.25}, "shared_object_id": 20}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 21}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 23}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 25}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 26}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 27}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 28}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 29}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 31}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 33}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 34}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 9, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 35}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 36}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 37}], "build_input_shape": {"weekday": {"class_name": "__tuple__", "items": [null, 1]}, "month": {"class_name": "__tuple__", "items": [null, 1]}, "commas": {"class_name": "__tuple__", "items": [null, 1]}, "exclam": {"class_name": "__tuple__", "items": [null, 1]}, "money": {"class_name": "__tuple__", "items": [null, 1]}, "dash": {"class_name": "__tuple__", "items": [null, 1]}, "colon": {"class_name": "__tuple__", "items": [null, 1]}, "parenthesis": {"class_name": "__tuple__", "items": [null, 1]}, "city_embed_1": {"class_name": "__tuple__", "items": [null, 1]}, "city_embed_2": {"class_name": "__tuple__", "items": [null, 1]}, "city_embed_3": {"class_name": "__tuple__", "items": [null, 1]}, "city_embed_4": {"class_name": "__tuple__", "items": [null, 1]}, "city_embed_5": {"class_name": "__tuple__", "items": [null, 1]}, "state_embed_1": {"class_name": "__tuple__", "items": [null, 1]}, "state_embed_2": {"class_name": "__tuple__", "items": [null, 1]}, "state_embed_3": {"class_name": "__tuple__", "items": [null, 1]}, "state_embed_4": {"class_name": "__tuple__", "items": [null, 1]}, "state_embed_5": {"class_name": "__tuple__", "items": [null, 1]}, "category_embed_1": {"class_name": "__tuple__", "items": [null, 1]}, "category_embed_2": {"class_name": "__tuple__", "items": [null, 1]}, "category_embed_3": {"class_name": "__tuple__", "items": [null, 1]}, "category_embed_4": {"class_name": "__tuple__", "items": [null, 1]}, "category_embed_5": {"class_name": "__tuple__", "items": [null, 1]}}}}, "training_config": {"loss": "loss_sparse_categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}, "shared_object_id": 39}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?1
_feature_columns

_resources
	variables
regularization_losses
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?0
_tf_keras_layer?0{"name": "dense_features_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "DenseFeatures", "config": {"name": "dense_features_1", "trainable": true, "dtype": "float32", "feature_columns": [{"class_name": "NumericColumn", "config": {"key": "category_embed_1", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "category_embed_2", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "category_embed_3", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "category_embed_4", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "category_embed_5", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "city_embed_1", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "city_embed_2", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "city_embed_3", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "city_embed_4", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "city_embed_5", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "colon", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "commas", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "dash", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "exclam", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "money", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "month", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "parenthesis", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "state_embed_1", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "state_embed_2", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "state_embed_3", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "state_embed_4", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "state_embed_5", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "weekday", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}], "partitioner": null}, "shared_object_id": 0, "build_input_shape": {"weekday": {"class_name": "TensorShape", "items": [null, 1]}, "month": {"class_name": "TensorShape", "items": [null, 1]}, "commas": {"class_name": "TensorShape", "items": [null, 1]}, "exclam": {"class_name": "TensorShape", "items": [null, 1]}, "money": {"class_name": "TensorShape", "items": [null, 1]}, "dash": {"class_name": "TensorShape", "items": [null, 1]}, "colon": {"class_name": "TensorShape", "items": [null, 1]}, "parenthesis": {"class_name": "TensorShape", "items": [null, 1]}, "city_embed_1": {"class_name": "TensorShape", "items": [null, 1]}, "city_embed_2": {"class_name": "TensorShape", "items": [null, 1]}, "city_embed_3": {"class_name": "TensorShape", "items": [null, 1]}, "city_embed_4": {"class_name": "TensorShape", "items": [null, 1]}, "city_embed_5": {"class_name": "TensorShape", "items": [null, 1]}, "state_embed_1": {"class_name": "TensorShape", "items": [null, 1]}, "state_embed_2": {"class_name": "TensorShape", "items": [null, 1]}, "state_embed_3": {"class_name": "TensorShape", "items": [null, 1]}, "state_embed_4": {"class_name": "TensorShape", "items": [null, 1]}, "state_embed_5": {"class_name": "TensorShape", "items": [null, 1]}, "category_embed_1": {"class_name": "TensorShape", "items": [null, 1]}, "category_embed_2": {"class_name": "TensorShape", "items": [null, 1]}, "category_embed_3": {"class_name": "TensorShape", "items": [null, 1]}, "category_embed_4": {"class_name": "TensorShape", "items": [null, 1]}, "category_embed_5": {"class_name": "TensorShape", "items": [null, 1]}}, "_is_feature_layer": true}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 23}}, "shared_object_id": 40}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 23]}}
?

axis
	gamma
 beta
!moving_mean
"moving_variance
#	variables
$regularization_losses
%trainable_variables
&	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "batch_normalization_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 32}}, "shared_object_id": 41}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?	

'kernel
(bias
)	variables
*regularization_losses
+trainable_variables
,	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.5}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 42}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?

-axis
	.gamma
/beta
0moving_mean
1moving_variance
2	variables
3regularization_losses
4trainable_variables
5	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "batch_normalization_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 14}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 16}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 128}}, "shared_object_id": 43}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?	

6kernel
7bias
8	variables
9regularization_losses
:trainable_variables
;	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.25}, "shared_object_id": 20}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 44}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?

<axis
	=gamma
>beta
?moving_mean
@moving_variance
A	variables
Bregularization_losses
Ctrainable_variables
D	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "batch_normalization_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 23}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 25}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 26, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 64}}, "shared_object_id": 45}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?

Ekernel
Fbias
G	variables
Hregularization_losses
Itrainable_variables
J	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 27}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 28}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 29, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 46}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?

Kaxis
	Lgamma
Mbeta
Nmoving_mean
Omoving_variance
P	variables
Qregularization_losses
Rtrainable_variables
S	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "batch_normalization_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 31}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 33}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 34, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 128}}, "shared_object_id": 47}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?

Tkernel
Ubias
V	variables
Wregularization_losses
Xtrainable_variables
Y	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 9, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 35}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 36}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 37, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 48}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?
Ziter

[beta_1

\beta_2
	]decay
^learning_ratem?m?m? m?'m?(m?.m?/m?6m?7m?=m?>m?Em?Fm?Lm?Mm?Tm?Um?v?v?v? v?'v?(v?.v?/v?6v?7v?=v?>v?Ev?Fv?Lv?Mv?Tv?Uv?"
	optimizer
 "
trackable_dict_wrapper
?
0
1
2
 3
!4
"5
'6
(7
.8
/9
010
111
612
713
=14
>15
?16
@17
E18
F19
L20
M21
N22
O23
T24
U25"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
0
1
2
 3
'4
(5
.6
/7
68
79
=10
>11
E12
F13
L14
M15
T16
U17"
trackable_list_wrapper
?
	variables
_metrics
`non_trainable_variables
regularization_losses

alayers
blayer_regularization_losses
clayer_metrics
trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
dmetrics
enon_trainable_variables
regularization_losses

flayers
glayer_regularization_losses
hlayer_metrics
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.:, 2sequential_1/dense_14/kernel
(:& 2sequential_1/dense_14/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
	variables
imetrics
jnon_trainable_variables
regularization_losses

klayers
llayer_regularization_losses
mlayer_metrics
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
7:5 2)sequential_1/batch_normalization_11/gamma
6:4 2(sequential_1/batch_normalization_11/beta
?:=  (2/sequential_1/batch_normalization_11/moving_mean
C:A  (23sequential_1/batch_normalization_11/moving_variance
<
0
 1
!2
"3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
?
#	variables
nmetrics
onon_trainable_variables
$regularization_losses

players
qlayer_regularization_losses
rlayer_metrics
%trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-	 ?2sequential_1/dense_13/kernel
):'?2sequential_1/dense_13/bias
.
'0
(1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
?
)	variables
smetrics
tnon_trainable_variables
*regularization_losses

ulayers
vlayer_regularization_losses
wlayer_metrics
+trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
8:6?2)sequential_1/batch_normalization_10/gamma
7:5?2(sequential_1/batch_normalization_10/beta
@:>? (2/sequential_1/batch_normalization_10/moving_mean
D:B? (23sequential_1/batch_normalization_10/moving_variance
<
.0
/1
02
13"
trackable_list_wrapper
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
?
2	variables
xmetrics
ynon_trainable_variables
3regularization_losses

zlayers
{layer_regularization_losses
|layer_metrics
4trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-	?@2sequential_1/dense_12/kernel
(:&@2sequential_1/dense_12/bias
.
60
71"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
?
8	variables
}metrics
~non_trainable_variables
9regularization_losses

layers
 ?layer_regularization_losses
?layer_metrics
:trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
6:4@2(sequential_1/batch_normalization_9/gamma
5:3@2'sequential_1/batch_normalization_9/beta
>:<@ (2.sequential_1/batch_normalization_9/moving_mean
B:@@ (22sequential_1/batch_normalization_9/moving_variance
<
=0
>1
?2
@3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
?
A	variables
?metrics
?non_trainable_variables
Bregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
Ctrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-	@?2sequential_1/dense_11/kernel
):'?2sequential_1/dense_11/bias
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
?
G	variables
?metrics
?non_trainable_variables
Hregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
Itrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
7:5?2(sequential_1/batch_normalization_8/gamma
6:4?2'sequential_1/batch_normalization_8/beta
?:=? (2.sequential_1/batch_normalization_8/moving_mean
C:A? (22sequential_1/batch_normalization_8/moving_variance
<
L0
M1
N2
O3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
?
P	variables
?metrics
?non_trainable_variables
Qregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
Rtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-	?	2sequential_1/dense_10/kernel
(:&	2sequential_1/dense_10/bias
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
?
V	variables
?metrics
?non_trainable_variables
Wregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
Xtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
0
?0
?1"
trackable_list_wrapper
X
!0
"1
02
13
?4
@5
N6
O7"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
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
.
!0
"1"
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
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
00
11"
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
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
?0
@1"
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
.
N0
O1"
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
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 49}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}, "shared_object_id": 39}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
3:1 2#Adam/sequential_1/dense_14/kernel/m
-:+ 2!Adam/sequential_1/dense_14/bias/m
<:: 20Adam/sequential_1/batch_normalization_11/gamma/m
;:9 2/Adam/sequential_1/batch_normalization_11/beta/m
4:2	 ?2#Adam/sequential_1/dense_13/kernel/m
.:,?2!Adam/sequential_1/dense_13/bias/m
=:;?20Adam/sequential_1/batch_normalization_10/gamma/m
<::?2/Adam/sequential_1/batch_normalization_10/beta/m
4:2	?@2#Adam/sequential_1/dense_12/kernel/m
-:+@2!Adam/sequential_1/dense_12/bias/m
;:9@2/Adam/sequential_1/batch_normalization_9/gamma/m
::8@2.Adam/sequential_1/batch_normalization_9/beta/m
4:2	@?2#Adam/sequential_1/dense_11/kernel/m
.:,?2!Adam/sequential_1/dense_11/bias/m
<::?2/Adam/sequential_1/batch_normalization_8/gamma/m
;:9?2.Adam/sequential_1/batch_normalization_8/beta/m
4:2	?	2#Adam/sequential_1/dense_10/kernel/m
-:+	2!Adam/sequential_1/dense_10/bias/m
3:1 2#Adam/sequential_1/dense_14/kernel/v
-:+ 2!Adam/sequential_1/dense_14/bias/v
<:: 20Adam/sequential_1/batch_normalization_11/gamma/v
;:9 2/Adam/sequential_1/batch_normalization_11/beta/v
4:2	 ?2#Adam/sequential_1/dense_13/kernel/v
.:,?2!Adam/sequential_1/dense_13/bias/v
=:;?20Adam/sequential_1/batch_normalization_10/gamma/v
<::?2/Adam/sequential_1/batch_normalization_10/beta/v
4:2	?@2#Adam/sequential_1/dense_12/kernel/v
-:+@2!Adam/sequential_1/dense_12/bias/v
;:9@2/Adam/sequential_1/batch_normalization_9/gamma/v
::8@2.Adam/sequential_1/batch_normalization_9/beta/v
4:2	@?2#Adam/sequential_1/dense_11/kernel/v
.:,?2!Adam/sequential_1/dense_11/bias/v
<::?2/Adam/sequential_1/batch_normalization_8/gamma/v
;:9?2.Adam/sequential_1/batch_normalization_8/beta/v
4:2	?	2#Adam/sequential_1/dense_10/kernel/v
-:+	2!Adam/sequential_1/dense_10/bias/v
?2?
-__inference_sequential_1_layer_call_fn_561573
-__inference_sequential_1_layer_call_fn_562605
-__inference_sequential_1_layer_call_fn_562684
-__inference_sequential_1_layer_call_fn_562227?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_sequential_1_layer_call_and_return_conditional_losses_563007
H__inference_sequential_1_layer_call_and_return_conditional_losses_563386
H__inference_sequential_1_layer_call_and_return_conditional_losses_562327
H__inference_sequential_1_layer_call_and_return_conditional_losses_562427?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
!__inference__wrapped_model_560459?

???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?	??	
?	??	
>
category_embed_1*?'
category_embed_1?????????
>
category_embed_2*?'
category_embed_2?????????
>
category_embed_3*?'
category_embed_3?????????
>
category_embed_4*?'
category_embed_4?????????
>
category_embed_5*?'
category_embed_5?????????
6
city_embed_1&?#
city_embed_1?????????
6
city_embed_2&?#
city_embed_2?????????
6
city_embed_3&?#
city_embed_3?????????
6
city_embed_4&?#
city_embed_4?????????
6
city_embed_5&?#
city_embed_5?????????
(
colon?
colon?????????
*
commas ?
commas?????????
&
dash?
dash?????????
*
exclam ?
exclam?????????
(
money?
money?????????
(
month?
month?????????
4
parenthesis%?"
parenthesis?????????
8
state_embed_1'?$
state_embed_1?????????
8
state_embed_2'?$
state_embed_2?????????
8
state_embed_3'?$
state_embed_3?????????
8
state_embed_4'?$
state_embed_4?????????
8
state_embed_5'?$
state_embed_5?????????
,
weekday!?
weekday?????????
?2?
1__inference_dense_features_1_layer_call_fn_563413
1__inference_dense_features_1_layer_call_fn_563440?
???
FullArgSpecE
args=?:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
L__inference_dense_features_1_layer_call_and_return_conditional_losses_563652
L__inference_dense_features_1_layer_call_and_return_conditional_losses_563864?
???
FullArgSpecE
args=?:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dense_14_layer_call_fn_563873?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_14_layer_call_and_return_conditional_losses_563884?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_batch_normalization_11_layer_call_fn_563897
7__inference_batch_normalization_11_layer_call_fn_563910?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_563930
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_563964?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dense_13_layer_call_fn_563979?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_13_layer_call_and_return_conditional_losses_563996?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_batch_normalization_10_layer_call_fn_564009
7__inference_batch_normalization_10_layer_call_fn_564022?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_564042
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_564076?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dense_12_layer_call_fn_564091?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_12_layer_call_and_return_conditional_losses_564108?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
6__inference_batch_normalization_9_layer_call_fn_564121
6__inference_batch_normalization_9_layer_call_fn_564134?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_564154
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_564188?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dense_11_layer_call_fn_564197?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_11_layer_call_and_return_conditional_losses_564208?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
6__inference_batch_normalization_8_layer_call_fn_564221
6__inference_batch_normalization_8_layer_call_fn_564234?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_564254
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_564288?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dense_10_layer_call_fn_564297?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_10_layer_call_and_return_conditional_losses_564308?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_564319?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_564330?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
$__inference_signature_wrapper_562526category_embed_1category_embed_2category_embed_3category_embed_4category_embed_5city_embed_1city_embed_2city_embed_3city_embed_4city_embed_5coloncommasdashexclammoneymonthparenthesisstate_embed_1state_embed_2state_embed_3state_embed_4state_embed_5weekday"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?

!__inference__wrapped_model_560459?
!" '(01/.67?@>=EFNOMLTU?	??	
?	??	
?	??	
>
category_embed_1*?'
category_embed_1?????????
>
category_embed_2*?'
category_embed_2?????????
>
category_embed_3*?'
category_embed_3?????????
>
category_embed_4*?'
category_embed_4?????????
>
category_embed_5*?'
category_embed_5?????????
6
city_embed_1&?#
city_embed_1?????????
6
city_embed_2&?#
city_embed_2?????????
6
city_embed_3&?#
city_embed_3?????????
6
city_embed_4&?#
city_embed_4?????????
6
city_embed_5&?#
city_embed_5?????????
(
colon?
colon?????????
*
commas ?
commas?????????
&
dash?
dash?????????
*
exclam ?
exclam?????????
(
money?
money?????????
(
month?
month?????????
4
parenthesis%?"
parenthesis?????????
8
state_embed_1'?$
state_embed_1?????????
8
state_embed_2'?$
state_embed_2?????????
8
state_embed_3'?$
state_embed_3?????????
8
state_embed_4'?$
state_embed_4?????????
8
state_embed_5'?$
state_embed_5?????????
,
weekday!?
weekday?????????
? "3?0
.
output_1"?
output_1?????????	?
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_564042d01/.4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_564076d01/.4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
7__inference_batch_normalization_10_layer_call_fn_564009W01/.4?1
*?'
!?
inputs??????????
p 
? "????????????
7__inference_batch_normalization_10_layer_call_fn_564022W01/.4?1
*?'
!?
inputs??????????
p
? "????????????
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_563930b!" 3?0
)?&
 ?
inputs????????? 
p 
? "%?"
?
0????????? 
? ?
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_563964b!" 3?0
)?&
 ?
inputs????????? 
p
? "%?"
?
0????????? 
? ?
7__inference_batch_normalization_11_layer_call_fn_563897U!" 3?0
)?&
 ?
inputs????????? 
p 
? "?????????? ?
7__inference_batch_normalization_11_layer_call_fn_563910U!" 3?0
)?&
 ?
inputs????????? 
p
? "?????????? ?
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_564254dNOML4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_564288dNOML4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
6__inference_batch_normalization_8_layer_call_fn_564221WNOML4?1
*?'
!?
inputs??????????
p 
? "????????????
6__inference_batch_normalization_8_layer_call_fn_564234WNOML4?1
*?'
!?
inputs??????????
p
? "????????????
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_564154b?@>=3?0
)?&
 ?
inputs?????????@
p 
? "%?"
?
0?????????@
? ?
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_564188b?@>=3?0
)?&
 ?
inputs?????????@
p
? "%?"
?
0?????????@
? ?
6__inference_batch_normalization_9_layer_call_fn_564121U?@>=3?0
)?&
 ?
inputs?????????@
p 
? "??????????@?
6__inference_batch_normalization_9_layer_call_fn_564134U?@>=3?0
)?&
 ?
inputs?????????@
p
? "??????????@?
D__inference_dense_10_layer_call_and_return_conditional_losses_564308]TU0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????	
? }
)__inference_dense_10_layer_call_fn_564297PTU0?-
&?#
!?
inputs??????????
? "??????????	?
D__inference_dense_11_layer_call_and_return_conditional_losses_564208]EF/?,
%?"
 ?
inputs?????????@
? "&?#
?
0??????????
? }
)__inference_dense_11_layer_call_fn_564197PEF/?,
%?"
 ?
inputs?????????@
? "????????????
D__inference_dense_12_layer_call_and_return_conditional_losses_564108]670?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? }
)__inference_dense_12_layer_call_fn_564091P670?-
&?#
!?
inputs??????????
? "??????????@?
D__inference_dense_13_layer_call_and_return_conditional_losses_563996]'(/?,
%?"
 ?
inputs????????? 
? "&?#
?
0??????????
? }
)__inference_dense_13_layer_call_fn_563979P'(/?,
%?"
 ?
inputs????????? 
? "????????????
D__inference_dense_14_layer_call_and_return_conditional_losses_563884\/?,
%?"
 ?
inputs?????????
? "%?"
?
0????????? 
? |
)__inference_dense_14_layer_call_fn_563873O/?,
%?"
 ?
inputs?????????
? "?????????? ?
L__inference_dense_features_1_layer_call_and_return_conditional_losses_563652????
???
???
G
category_embed_13?0
features/category_embed_1?????????
G
category_embed_23?0
features/category_embed_2?????????
G
category_embed_33?0
features/category_embed_3?????????
G
category_embed_43?0
features/category_embed_4?????????
G
category_embed_53?0
features/category_embed_5?????????
?
city_embed_1/?,
features/city_embed_1?????????
?
city_embed_2/?,
features/city_embed_2?????????
?
city_embed_3/?,
features/city_embed_3?????????
?
city_embed_4/?,
features/city_embed_4?????????
?
city_embed_5/?,
features/city_embed_5?????????
1
colon(?%
features/colon?????????
3
commas)?&
features/commas?????????
/
dash'?$
features/dash?????????
3
exclam)?&
features/exclam?????????
1
money(?%
features/money?????????
1
month(?%
features/month?????????
=
parenthesis.?+
features/parenthesis?????????
A
state_embed_10?-
features/state_embed_1?????????
A
state_embed_20?-
features/state_embed_2?????????
A
state_embed_30?-
features/state_embed_3?????????
A
state_embed_40?-
features/state_embed_4?????????
A
state_embed_50?-
features/state_embed_5?????????
5
weekday*?'
features/weekday?????????

 
p 
? "%?"
?
0?????????
? ?
L__inference_dense_features_1_layer_call_and_return_conditional_losses_563864????
???
???
G
category_embed_13?0
features/category_embed_1?????????
G
category_embed_23?0
features/category_embed_2?????????
G
category_embed_33?0
features/category_embed_3?????????
G
category_embed_43?0
features/category_embed_4?????????
G
category_embed_53?0
features/category_embed_5?????????
?
city_embed_1/?,
features/city_embed_1?????????
?
city_embed_2/?,
features/city_embed_2?????????
?
city_embed_3/?,
features/city_embed_3?????????
?
city_embed_4/?,
features/city_embed_4?????????
?
city_embed_5/?,
features/city_embed_5?????????
1
colon(?%
features/colon?????????
3
commas)?&
features/commas?????????
/
dash'?$
features/dash?????????
3
exclam)?&
features/exclam?????????
1
money(?%
features/money?????????
1
month(?%
features/month?????????
=
parenthesis.?+
features/parenthesis?????????
A
state_embed_10?-
features/state_embed_1?????????
A
state_embed_20?-
features/state_embed_2?????????
A
state_embed_30?-
features/state_embed_3?????????
A
state_embed_40?-
features/state_embed_4?????????
A
state_embed_50?-
features/state_embed_5?????????
5
weekday*?'
features/weekday?????????

 
p
? "%?"
?
0?????????
? ?
1__inference_dense_features_1_layer_call_fn_563413????
???
???
G
category_embed_13?0
features/category_embed_1?????????
G
category_embed_23?0
features/category_embed_2?????????
G
category_embed_33?0
features/category_embed_3?????????
G
category_embed_43?0
features/category_embed_4?????????
G
category_embed_53?0
features/category_embed_5?????????
?
city_embed_1/?,
features/city_embed_1?????????
?
city_embed_2/?,
features/city_embed_2?????????
?
city_embed_3/?,
features/city_embed_3?????????
?
city_embed_4/?,
features/city_embed_4?????????
?
city_embed_5/?,
features/city_embed_5?????????
1
colon(?%
features/colon?????????
3
commas)?&
features/commas?????????
/
dash'?$
features/dash?????????
3
exclam)?&
features/exclam?????????
1
money(?%
features/money?????????
1
month(?%
features/month?????????
=
parenthesis.?+
features/parenthesis?????????
A
state_embed_10?-
features/state_embed_1?????????
A
state_embed_20?-
features/state_embed_2?????????
A
state_embed_30?-
features/state_embed_3?????????
A
state_embed_40?-
features/state_embed_4?????????
A
state_embed_50?-
features/state_embed_5?????????
5
weekday*?'
features/weekday?????????

 
p 
? "???????????
1__inference_dense_features_1_layer_call_fn_563440????
???
???
G
category_embed_13?0
features/category_embed_1?????????
G
category_embed_23?0
features/category_embed_2?????????
G
category_embed_33?0
features/category_embed_3?????????
G
category_embed_43?0
features/category_embed_4?????????
G
category_embed_53?0
features/category_embed_5?????????
?
city_embed_1/?,
features/city_embed_1?????????
?
city_embed_2/?,
features/city_embed_2?????????
?
city_embed_3/?,
features/city_embed_3?????????
?
city_embed_4/?,
features/city_embed_4?????????
?
city_embed_5/?,
features/city_embed_5?????????
1
colon(?%
features/colon?????????
3
commas)?&
features/commas?????????
/
dash'?$
features/dash?????????
3
exclam)?&
features/exclam?????????
1
money(?%
features/money?????????
1
month(?%
features/month?????????
=
parenthesis.?+
features/parenthesis?????????
A
state_embed_10?-
features/state_embed_1?????????
A
state_embed_20?-
features/state_embed_2?????????
A
state_embed_30?-
features/state_embed_3?????????
A
state_embed_40?-
features/state_embed_4?????????
A
state_embed_50?-
features/state_embed_5?????????
5
weekday*?'
features/weekday?????????

 
p
? "??????????;
__inference_loss_fn_0_564319'?

? 
? "? ;
__inference_loss_fn_1_5643306?

? 
? "? ?
H__inference_sequential_1_layer_call_and_return_conditional_losses_562327?
!" '(01/.67?@>=EFNOMLTU?	??	
?	??	
?	??	
>
category_embed_1*?'
category_embed_1?????????
>
category_embed_2*?'
category_embed_2?????????
>
category_embed_3*?'
category_embed_3?????????
>
category_embed_4*?'
category_embed_4?????????
>
category_embed_5*?'
category_embed_5?????????
6
city_embed_1&?#
city_embed_1?????????
6
city_embed_2&?#
city_embed_2?????????
6
city_embed_3&?#
city_embed_3?????????
6
city_embed_4&?#
city_embed_4?????????
6
city_embed_5&?#
city_embed_5?????????
(
colon?
colon?????????
*
commas ?
commas?????????
&
dash?
dash?????????
*
exclam ?
exclam?????????
(
money?
money?????????
(
month?
month?????????
4
parenthesis%?"
parenthesis?????????
8
state_embed_1'?$
state_embed_1?????????
8
state_embed_2'?$
state_embed_2?????????
8
state_embed_3'?$
state_embed_3?????????
8
state_embed_4'?$
state_embed_4?????????
8
state_embed_5'?$
state_embed_5?????????
,
weekday!?
weekday?????????
p 

 
? "%?"
?
0?????????	
? ?
H__inference_sequential_1_layer_call_and_return_conditional_losses_562427?
!" '(01/.67?@>=EFNOMLTU?	??	
?	??	
?	??	
>
category_embed_1*?'
category_embed_1?????????
>
category_embed_2*?'
category_embed_2?????????
>
category_embed_3*?'
category_embed_3?????????
>
category_embed_4*?'
category_embed_4?????????
>
category_embed_5*?'
category_embed_5?????????
6
city_embed_1&?#
city_embed_1?????????
6
city_embed_2&?#
city_embed_2?????????
6
city_embed_3&?#
city_embed_3?????????
6
city_embed_4&?#
city_embed_4?????????
6
city_embed_5&?#
city_embed_5?????????
(
colon?
colon?????????
*
commas ?
commas?????????
&
dash?
dash?????????
*
exclam ?
exclam?????????
(
money?
money?????????
(
month?
month?????????
4
parenthesis%?"
parenthesis?????????
8
state_embed_1'?$
state_embed_1?????????
8
state_embed_2'?$
state_embed_2?????????
8
state_embed_3'?$
state_embed_3?????????
8
state_embed_4'?$
state_embed_4?????????
8
state_embed_5'?$
state_embed_5?????????
,
weekday!?
weekday?????????
p

 
? "%?"
?
0?????????	
? ?
H__inference_sequential_1_layer_call_and_return_conditional_losses_563007?!" '(01/.67?@>=EFNOMLTU???
???
???

E
category_embed_11?.
inputs/category_embed_1?????????
E
category_embed_21?.
inputs/category_embed_2?????????
E
category_embed_31?.
inputs/category_embed_3?????????
E
category_embed_41?.
inputs/category_embed_4?????????
E
category_embed_51?.
inputs/category_embed_5?????????
=
city_embed_1-?*
inputs/city_embed_1?????????
=
city_embed_2-?*
inputs/city_embed_2?????????
=
city_embed_3-?*
inputs/city_embed_3?????????
=
city_embed_4-?*
inputs/city_embed_4?????????
=
city_embed_5-?*
inputs/city_embed_5?????????
/
colon&?#
inputs/colon?????????
1
commas'?$
inputs/commas?????????
-
dash%?"
inputs/dash?????????
1
exclam'?$
inputs/exclam?????????
/
money&?#
inputs/money?????????
/
month&?#
inputs/month?????????
;
parenthesis,?)
inputs/parenthesis?????????
?
state_embed_1.?+
inputs/state_embed_1?????????
?
state_embed_2.?+
inputs/state_embed_2?????????
?
state_embed_3.?+
inputs/state_embed_3?????????
?
state_embed_4.?+
inputs/state_embed_4?????????
?
state_embed_5.?+
inputs/state_embed_5?????????
3
weekday(?%
inputs/weekday?????????
p 

 
? "%?"
?
0?????????	
? ?
H__inference_sequential_1_layer_call_and_return_conditional_losses_563386?!" '(01/.67?@>=EFNOMLTU???
???
???

E
category_embed_11?.
inputs/category_embed_1?????????
E
category_embed_21?.
inputs/category_embed_2?????????
E
category_embed_31?.
inputs/category_embed_3?????????
E
category_embed_41?.
inputs/category_embed_4?????????
E
category_embed_51?.
inputs/category_embed_5?????????
=
city_embed_1-?*
inputs/city_embed_1?????????
=
city_embed_2-?*
inputs/city_embed_2?????????
=
city_embed_3-?*
inputs/city_embed_3?????????
=
city_embed_4-?*
inputs/city_embed_4?????????
=
city_embed_5-?*
inputs/city_embed_5?????????
/
colon&?#
inputs/colon?????????
1
commas'?$
inputs/commas?????????
-
dash%?"
inputs/dash?????????
1
exclam'?$
inputs/exclam?????????
/
money&?#
inputs/money?????????
/
month&?#
inputs/month?????????
;
parenthesis,?)
inputs/parenthesis?????????
?
state_embed_1.?+
inputs/state_embed_1?????????
?
state_embed_2.?+
inputs/state_embed_2?????????
?
state_embed_3.?+
inputs/state_embed_3?????????
?
state_embed_4.?+
inputs/state_embed_4?????????
?
state_embed_5.?+
inputs/state_embed_5?????????
3
weekday(?%
inputs/weekday?????????
p

 
? "%?"
?
0?????????	
? ?

-__inference_sequential_1_layer_call_fn_561573?
!" '(01/.67?@>=EFNOMLTU?	??	
?	??	
?	??	
>
category_embed_1*?'
category_embed_1?????????
>
category_embed_2*?'
category_embed_2?????????
>
category_embed_3*?'
category_embed_3?????????
>
category_embed_4*?'
category_embed_4?????????
>
category_embed_5*?'
category_embed_5?????????
6
city_embed_1&?#
city_embed_1?????????
6
city_embed_2&?#
city_embed_2?????????
6
city_embed_3&?#
city_embed_3?????????
6
city_embed_4&?#
city_embed_4?????????
6
city_embed_5&?#
city_embed_5?????????
(
colon?
colon?????????
*
commas ?
commas?????????
&
dash?
dash?????????
*
exclam ?
exclam?????????
(
money?
money?????????
(
month?
month?????????
4
parenthesis%?"
parenthesis?????????
8
state_embed_1'?$
state_embed_1?????????
8
state_embed_2'?$
state_embed_2?????????
8
state_embed_3'?$
state_embed_3?????????
8
state_embed_4'?$
state_embed_4?????????
8
state_embed_5'?$
state_embed_5?????????
,
weekday!?
weekday?????????
p 

 
? "??????????	?

-__inference_sequential_1_layer_call_fn_562227?
!" '(01/.67?@>=EFNOMLTU?	??	
?	??	
?	??	
>
category_embed_1*?'
category_embed_1?????????
>
category_embed_2*?'
category_embed_2?????????
>
category_embed_3*?'
category_embed_3?????????
>
category_embed_4*?'
category_embed_4?????????
>
category_embed_5*?'
category_embed_5?????????
6
city_embed_1&?#
city_embed_1?????????
6
city_embed_2&?#
city_embed_2?????????
6
city_embed_3&?#
city_embed_3?????????
6
city_embed_4&?#
city_embed_4?????????
6
city_embed_5&?#
city_embed_5?????????
(
colon?
colon?????????
*
commas ?
commas?????????
&
dash?
dash?????????
*
exclam ?
exclam?????????
(
money?
money?????????
(
month?
month?????????
4
parenthesis%?"
parenthesis?????????
8
state_embed_1'?$
state_embed_1?????????
8
state_embed_2'?$
state_embed_2?????????
8
state_embed_3'?$
state_embed_3?????????
8
state_embed_4'?$
state_embed_4?????????
8
state_embed_5'?$
state_embed_5?????????
,
weekday!?
weekday?????????
p

 
? "??????????	?
-__inference_sequential_1_layer_call_fn_562605?!" '(01/.67?@>=EFNOMLTU???
???
???

E
category_embed_11?.
inputs/category_embed_1?????????
E
category_embed_21?.
inputs/category_embed_2?????????
E
category_embed_31?.
inputs/category_embed_3?????????
E
category_embed_41?.
inputs/category_embed_4?????????
E
category_embed_51?.
inputs/category_embed_5?????????
=
city_embed_1-?*
inputs/city_embed_1?????????
=
city_embed_2-?*
inputs/city_embed_2?????????
=
city_embed_3-?*
inputs/city_embed_3?????????
=
city_embed_4-?*
inputs/city_embed_4?????????
=
city_embed_5-?*
inputs/city_embed_5?????????
/
colon&?#
inputs/colon?????????
1
commas'?$
inputs/commas?????????
-
dash%?"
inputs/dash?????????
1
exclam'?$
inputs/exclam?????????
/
money&?#
inputs/money?????????
/
month&?#
inputs/month?????????
;
parenthesis,?)
inputs/parenthesis?????????
?
state_embed_1.?+
inputs/state_embed_1?????????
?
state_embed_2.?+
inputs/state_embed_2?????????
?
state_embed_3.?+
inputs/state_embed_3?????????
?
state_embed_4.?+
inputs/state_embed_4?????????
?
state_embed_5.?+
inputs/state_embed_5?????????
3
weekday(?%
inputs/weekday?????????
p 

 
? "??????????	?
-__inference_sequential_1_layer_call_fn_562684?!" '(01/.67?@>=EFNOMLTU???
???
???

E
category_embed_11?.
inputs/category_embed_1?????????
E
category_embed_21?.
inputs/category_embed_2?????????
E
category_embed_31?.
inputs/category_embed_3?????????
E
category_embed_41?.
inputs/category_embed_4?????????
E
category_embed_51?.
inputs/category_embed_5?????????
=
city_embed_1-?*
inputs/city_embed_1?????????
=
city_embed_2-?*
inputs/city_embed_2?????????
=
city_embed_3-?*
inputs/city_embed_3?????????
=
city_embed_4-?*
inputs/city_embed_4?????????
=
city_embed_5-?*
inputs/city_embed_5?????????
/
colon&?#
inputs/colon?????????
1
commas'?$
inputs/commas?????????
-
dash%?"
inputs/dash?????????
1
exclam'?$
inputs/exclam?????????
/
money&?#
inputs/money?????????
/
month&?#
inputs/month?????????
;
parenthesis,?)
inputs/parenthesis?????????
?
state_embed_1.?+
inputs/state_embed_1?????????
?
state_embed_2.?+
inputs/state_embed_2?????????
?
state_embed_3.?+
inputs/state_embed_3?????????
?
state_embed_4.?+
inputs/state_embed_4?????????
?
state_embed_5.?+
inputs/state_embed_5?????????
3
weekday(?%
inputs/weekday?????????
p

 
? "??????????	?

$__inference_signature_wrapper_562526?
!" '(01/.67?@>=EFNOMLTU?	??	
? 
?	??	
>
category_embed_1*?'
category_embed_1?????????
>
category_embed_2*?'
category_embed_2?????????
>
category_embed_3*?'
category_embed_3?????????
>
category_embed_4*?'
category_embed_4?????????
>
category_embed_5*?'
category_embed_5?????????
6
city_embed_1&?#
city_embed_1?????????
6
city_embed_2&?#
city_embed_2?????????
6
city_embed_3&?#
city_embed_3?????????
6
city_embed_4&?#
city_embed_4?????????
6
city_embed_5&?#
city_embed_5?????????
(
colon?
colon?????????
*
commas ?
commas?????????
&
dash?
dash?????????
*
exclam ?
exclam?????????
(
money?
money?????????
(
month?
month?????????
4
parenthesis%?"
parenthesis?????????
8
state_embed_1'?$
state_embed_1?????????
8
state_embed_2'?$
state_embed_2?????????
8
state_embed_3'?$
state_embed_3?????????
8
state_embed_4'?$
state_embed_4?????????
8
state_embed_5'?$
state_embed_5?????????
,
weekday!?
weekday?????????"3?0
.
output_1"?
output_1?????????	