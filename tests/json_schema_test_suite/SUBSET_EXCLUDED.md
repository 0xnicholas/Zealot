# draft7 子集纳入记录（共 151 用例）


## additionalProperties.json（含 1 / 排除 6）

- 排除 [additionalProperties being false does not allow other properties] ALL：compile: JSON schema compile error: 关键字 `patternProperties` 暂不支持（P2 显式报错，不静默忽略）
- 排除 [non-ASCII pattern with additionalProperties] ALL：compile: JSON schema compile error: 关键字 `patternProperties` 暂不支持（P2 显式报错，不静默忽略）
- 排除 [additionalProperties with schema] ALL：compile: JSON schema compile error: additionalProperties 仅支持布尔形态（schema 形态暂不支持）
- 排除 [additionalProperties can exist by itself] ALL：compile: JSON schema compile error: additionalProperties 仅支持布尔形态（schema 形态暂不支持）
- 排除 [additionalProperties does not look in applicators] ALL：compile: JSON schema compile error: 关键字 `allOf` 暂不支持（P2 显式报错，不静默忽略）
- 排除 [additionalProperties with null valued instance properties] ALL：compile: JSON schema compile error: additionalProperties 仅支持布尔形态（schema 形态暂不支持）

## anyOf.json（含 10 / 排除 3）

- 排除 [anyOf] ALL：compile: JSON schema compile error: 关键字 minimum 需与 type: integer 搭配（无类型 schema 无法约束，暂不支持静默忽略）
- 排除 [anyOf with base schema] ALL：compile: JSON schema compile error: type 与 anyOf/oneOf 组合（交集语义）暂不支持
- 排除 [anyOf complex types] second anyOf valid (complex)：semantics: 期望 valid=True 实际=False (data={"foo": "baz"})

## enum.json（含 41 / 排除 4）

- 排除 [enum with 0 does not match false] float zero is valid：semantics: 期望 valid=True 实际=False (data=0.0)
- 排除 [enum with [0] does not match [false]] [0.0] is valid：semantics: 期望 valid=True 实际=False (data=[0.0])
- 排除 [enum with 1 does not match true] float one is valid：semantics: 期望 valid=True 实际=False (data=1.0)
- 排除 [enum with [1] does not match [true]] [1.0] is valid：semantics: 期望 valid=True 实际=False (data=[1.0])

## exclusiveMaximum.json（含 0 / 排除 1）

- 排除 [exclusiveMaximum validation] ALL：compile: JSON schema compile error: 关键字 exclusiveMaximum 需与 type: integer 搭配（无类型 schema 无法约束，暂不支持静默忽略）

## exclusiveMinimum.json（含 0 / 排除 1）

- 排除 [exclusiveMinimum validation] ALL：compile: JSON schema compile error: 关键字 exclusiveMinimum 需与 type: integer 搭配（无类型 schema 无法约束，暂不支持静默忽略）

## items.json（含 3 / 排除 8）

- 排除 [a schema given for items] ALL：compile: JSON schema compile error: 关键字 items 需与 type: array 搭配（无类型 schema 无法约束，暂不支持静默忽略）
- 排除 [an array of schemas for items] ALL：compile: JSON schema compile error: items 仅支持单 schema 对象形态（tuple 数组/布尔形态暂不支持）
- 排除 [items with boolean schema (true)] ALL：compile: JSON schema compile error: items 仅支持单 schema 对象形态（tuple 数组/布尔形态暂不支持）
- 排除 [items with boolean schema (false)] ALL：compile: JSON schema compile error: items 仅支持单 schema 对象形态（tuple 数组/布尔形态暂不支持）
- 排除 [items with boolean schemas] ALL：compile: JSON schema compile error: items 仅支持单 schema 对象形态（tuple 数组/布尔形态暂不支持）
- 排除 [items and subitems] ALL：compile: JSON schema compile error: items 仅支持单 schema 对象形态（tuple 数组/布尔形态暂不支持）
- 排除 [single-form items with null instance elements] ALL：compile: JSON schema compile error: 关键字 items 需与 type: array 搭配（无类型 schema 无法约束，暂不支持静默忽略）
- 排除 [array-form items with null instance elements] ALL：compile: JSON schema compile error: items 仅支持单 schema 对象形态（tuple 数组/布尔形态暂不支持）

## maxItems.json（含 0 / 排除 2）

- 排除 [maxItems validation] ALL：compile: JSON schema compile error: 关键字 maxItems 需与 type: array 搭配（无类型 schema 无法约束，暂不支持静默忽略）
- 排除 [maxItems validation with a decimal] ALL：compile: JSON schema compile error: 关键字 maxItems 需与 type: array 搭配（无类型 schema 无法约束，暂不支持静默忽略）

## maxLength.json（含 0 / 排除 2）

- 排除 [maxLength validation] ALL：compile: JSON schema compile error: 关键字 maxLength 需与 type: string 搭配（无类型 schema 无法约束，暂不支持静默忽略）
- 排除 [maxLength validation with a decimal] ALL：compile: JSON schema compile error: 关键字 maxLength 需与 type: string 搭配（无类型 schema 无法约束，暂不支持静默忽略）

## maximum.json（含 0 / 排除 2）

- 排除 [maximum validation] ALL：compile: JSON schema compile error: 关键字 maximum 需与 type: integer 搭配（无类型 schema 无法约束，暂不支持静默忽略）
- 排除 [maximum validation with unsigned integer] ALL：compile: JSON schema compile error: 关键字 maximum 需与 type: integer 搭配（无类型 schema 无法约束，暂不支持静默忽略）

## minItems.json（含 0 / 排除 2）

- 排除 [minItems validation] ALL：compile: JSON schema compile error: 关键字 minItems 需与 type: array 搭配（无类型 schema 无法约束，暂不支持静默忽略）
- 排除 [minItems validation with a decimal] ALL：compile: JSON schema compile error: 关键字 minItems 需与 type: array 搭配（无类型 schema 无法约束，暂不支持静默忽略）

## minLength.json（含 0 / 排除 2）

- 排除 [minLength validation] ALL：compile: JSON schema compile error: 关键字 minLength 需与 type: string 搭配（无类型 schema 无法约束，暂不支持静默忽略）
- 排除 [minLength validation with a decimal] ALL：compile: JSON schema compile error: 关键字 minLength 需与 type: string 搭配（无类型 schema 无法约束，暂不支持静默忽略）

## minimum.json（含 0 / 排除 2）

- 排除 [minimum validation] ALL：compile: JSON schema compile error: 关键字 minimum 需与 type: integer 搭配（无类型 schema 无法约束，暂不支持静默忽略）
- 排除 [minimum validation with signed integer] ALL：compile: JSON schema compile error: 关键字 minimum 需与 type: integer 搭配（无类型 schema 无法约束，暂不支持静默忽略）

## multipleOf.json（含 0 / 排除 5）

- 排除 [by int] ALL：compile: JSON schema compile error: 关键字 multipleOf 需与 type: integer 搭配（无类型 schema 无法约束，暂不支持静默忽略）
- 排除 [by number] ALL：compile: JSON schema compile error: 关键字 multipleOf 需与 type: integer 搭配（无类型 schema 无法约束，暂不支持静默忽略）
- 排除 [by small number] ALL：compile: JSON schema compile error: 关键字 multipleOf 需与 type: integer 搭配（无类型 schema 无法约束，暂不支持静默忽略）
- 排除 [float division = inf] ALL：compile: JSON schema compile error: multipleOf 小数位数过多（>6 位十进制）或不可精确表示
- 排除 [small multiple of large integer] ALL：compile: JSON schema compile error: multipleOf 小数位数过多（>6 位十进制）或不可精确表示

## oneOf.json（含 9 / 排除 10）

- 排除 [oneOf] ALL：compile: JSON schema compile error: 关键字 minimum 需与 type: integer 搭配（无类型 schema 无法约束，暂不支持静默忽略）
- 排除 [oneOf with base schema] ALL：compile: JSON schema compile error: type 与 anyOf/oneOf 组合（交集语义）暂不支持
- 排除 [oneOf with boolean schemas, all true] any value is invalid：semantics: 期望 valid=False 实际=True (data="foo")
- 排除 [oneOf with boolean schemas, more than one true] any value is invalid：semantics: 期望 valid=False 实际=True (data="foo")
- 排除 [oneOf complex types] second oneOf valid (complex)：semantics: 期望 valid=True 实际=False (data={"foo": "baz"})
- 排除 [oneOf complex types] both oneOf valid (complex)：semantics: 期望 valid=False 实际=True (data={"foo": "baz", "bar": 2})
- 排除 [oneOf with empty schema] both valid - invalid：semantics: 期望 valid=False 实际=True (data=123)
- 排除 [oneOf with required] ALL：compile: JSON schema compile error: type 与 anyOf/oneOf 组合（交集语义）暂不支持
- 排除 [oneOf with missing optional property] second oneOf valid：semantics: 期望 valid=True 实际=False (data={"foo": "foo"})
- 排除 [oneOf with missing optional property] both oneOf valid：semantics: 期望 valid=False 实际=True (data={"foo": "foo", "bar": 8})

## pattern.json（含 0 / 排除 2）

- 排除 [pattern validation] ALL：compile: JSON schema compile error: 关键字 pattern 需与 type: string 搭配（无类型 schema 无法约束，暂不支持静默忽略）
- 排除 [pattern is not anchored] ALL：compile: JSON schema compile error: 关键字 pattern 需与 type: string 搭配（无类型 schema 无法约束，暂不支持静默忽略）

## properties.json（含 16 / 排除 5）

- 排除 [object properties validation] ignores arrays：semantics: 期望 valid=True 实际=False (data=[])
- 排除 [object properties validation] ignores other non-objects：semantics: 期望 valid=True 实际=False (data=12)
- 排除 [properties, patternProperties, additionalProperties interaction] ALL：compile: JSON schema compile error: 关键字 `patternProperties` 暂不支持（P2 显式报错，不静默忽略）
- 排除 [properties whose names are Javascript object property names] ignores arrays：semantics: 期望 valid=True 实际=False (data=[])
- 排除 [properties whose names are Javascript object property names] ignores other non-objects：semantics: 期望 valid=True 实际=False (data=12)

## required.json（含 11 / 排除 7）

- 排除 [required validation] ignores arrays：semantics: 期望 valid=True 实际=False (data=[])
- 排除 [required validation] ignores strings：semantics: 期望 valid=True 实际=False (data="")
- 排除 [required validation] ignores other non-objects：semantics: 期望 valid=True 实际=False (data=12)
- 排除 [required validation] ignores null：semantics: 期望 valid=True 实际=False (data=null)
- 排除 [required validation] ignores boolean：semantics: 期望 valid=True 实际=False (data=true)
- 排除 [required properties whose names are Javascript object property names] ignores arrays：semantics: 期望 valid=True 实际=False (data=[])
- 排除 [required properties whose names are Javascript object property names] ignores other non-objects：semantics: 期望 valid=True 实际=False (data=12)

## type.json（含 60 / 排除 5）

- 排除 [integer type matches integers] a float with zero fractional part is an integer：semantics: 期望 valid=True 实际=False (data=1.0)
- 排除 [multiple types can be specified in an array] ALL：compile: JSON schema compile error: type 仅支持单字符串形态（数组/布尔/未知类型暂不支持）
- 排除 [type as array with one item] ALL：compile: JSON schema compile error: type 仅支持单字符串形态（数组/布尔/未知类型暂不支持）
- 排除 [type: array or object] ALL：compile: JSON schema compile error: type 仅支持单字符串形态（数组/布尔/未知类型暂不支持）
- 排除 [type: array, object or null] ALL：compile: JSON schema compile error: type 仅支持单字符串形态（数组/布尔/未知类型暂不支持）
