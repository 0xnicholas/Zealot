# JSON Schema Test Suite（draft-07）子集回归记录（M5-⑩, issue #24）

## 方法

- 来源：[json-schema-org/JSON-Schema-Test-Suite](https://github.com/json-schema-org/JSON-Schema-Test-Suite) `tests/draft7/` 18 个文件（`draft7/` 目录原样保存，作再生参考）。
- 生成器：对每个 case 编译（compile）→ 数据字节走自动机（walk）→ 与套件 `valid` 标志比对。
- **纳入**：编译成功且 walk 结果与套件一致的用例 → `subset/`（151 用例 / 9 文件）。
- **排除**：`SUBSET_EXCLUDED.md` 逐条记录原因（编译错误原文 / 语义差异期望 vs 实际）。
- 回归：`tests/test_schema_suite.py` 对 `subset/` 全部用例断言编译成功 + walk 与套件一致。
- 复现：`/tmp/gen_suite_subset.py`（编译后扩展的生成脚本，含编译错误原文）。

## 纳入的文件（151 用例）

| 文件 | 用例 | 覆盖 |
|---|---|---|
| type.json | 60 | 六种基础类型（除 `1.0` 计整数等 5 项差异） |
| enum.json | 41 | 任意值精确匹配（含转义/控制字符成员） |
| properties.json | 16 | 属性约束（含转义键名） |
| required.json | 11 | required 门（含转义键/`__proto__` 等 JS 键名） |
| anyOf.json | 10 | 并集语义 |
| oneOf.json | 9 | 并集语义（恰一不强制） |
| items.json | 3 | 单 schema items（typed） |
| additionalProperties.json | 1 | 默认 true + 无 properties |

## 排除类别（全部为设计性差异，非缺陷）

1. **无 type 的约束关键字**（minimum/maximum/exclusive*/multipleOf/minLength/maxLength/
   minItems/maxItems/pattern/format/items）——编译报错「需与 type: X 搭配」。生成引擎
   无法对无类型 schema 施加约束，不静默降级（unsupported > unsound，issue #23 延续）。
   等价的 typed 约束由 matcher/schema 单测与端到端用例覆盖（M5-⑤⑥⑦⑧）。
2. **P2 显式报错关键字**（patternProperties/allOf/含 schema 形态 additionalProperties/
   type 数组形态/items tuple 形态）——issue #23 守卫。
3. **properties/required 无 type 按 object 推断**：套件「ignores non-objects/arrays」类
   用例（非对象数据对对象 schema 判 valid）不适用——生成引擎把此类 schema 视为
   对象约束。
4. **数值表示等价**：`0.0` vs `0`、`1.0` vs `1`（enum/type integer）——引擎按字面表示
   匹配（生成侧不会产出 `1.0` 形式的整数，差异安全）。
5. **oneOf = 并集 + 同首字节分支取先匹配**：恰一（exactly-one）语义不强制；同首字节
   分支（如两个 object 分支）只走首个可接受分支（生成结果始终在并集内，sound；
   只是部分合法值不可达——文档化已知限制）。
6. **pattern 锚定语义**：套件「pattern is not anchored」断言子串匹配——本引擎按
   issue #22 决议采用整串锚定（llama.cpp 惯例）。
7. **multipleOf 数值上限**：小数位数 >6 或不可精确表示 → 显式报错（M5-⑤）。

## 套件暴露并修复的缺陷（本票价值）

- Any 值「字节汤」吞掉 `,`/`}`/`]` → 绕过 required/minItems 门（`{"bar":1}` 在
  required foo 下被接受）→ 重构为「结构骨架 + 2-环」任意 JSON 自动机
  （`build_any_json`，顶层/对象/数组三套叶子，编译无递归）。
- additionalProperties=true 无任意键路径（额外键被拒）→ 键汤 + 任意键分支
  （含键链中途折入，前缀冲突键可达）。
- 键链/enum 成员用原始字节 → 转义键名（`foo\nbar`、`foo"bar`）与含控制字符的
  enum 成员不可达 → 统一 serde_json 序列化字节。
- `{"required": [...]}` 无 type/properties → 静默降级 Any → 按 object 推断；
  required 键未在 properties 声明时补链（值 = 任意 JSON）。
