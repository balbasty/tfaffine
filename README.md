# tfaffine
Affine matrices encoded in their Lie algebra, in tensorflow

# demo

```python
>> import tensorflow as tf
>> import tfaffine as aff
>>
>> # build the affine basis
>> B = aff.affine_basis(3, 'Aff+')
>> B.shape
TensorShape([12, 4, 4])
>> # -> encodes 12-parameters affine matrices
>> 
>> # generate parameters (batch size = 2)
>> p = tf.random.normal((2, 12), dtype=tf.float32)
>>
>> # exponentiate
>> A = aff.affine_exp(p, B)
>> A.shape
TensorShape([2, 4, 4])
```

# dependencies
- `tensorflow`
