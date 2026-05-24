from pathlib import Path
import pandas as pd


samples = [
    # 正面 1
    ("这家店味道很好，下次还会再来", 1),
    ("服务员态度很热情，上菜速度也快", 1),
    ("包装很严实，收到的时候一点都没有损坏", 1),
    ("这个产品质量不错，和描述基本一致", 1),
    ("物流比预计快很多，体验很好", 1),
    ("房间很干净，晚上也比较安静", 1),
    ("虽然价格不低，但整体体验值得", 1),
    ("客服回复很及时，问题解决得很快", 1),
    ("这部电影节奏很好，剧情也很打动人", 1),
    ("老师讲得很清楚，例子也容易理解", 1),
    ("课程内容比较系统，适合入门学习", 1),
    ("APP界面简洁，功能也比较顺手", 1),
    ("耳机音质比想象中好，佩戴也舒服", 1),
    ("这次购物体验不错，售后也很负责", 1),
    ("虽然等了一会儿，但是菜品味道确实不错", 1),
    ("酒店位置方便，出门就是地铁站", 1),
    ("衣服面料舒服，尺码也比较合适", 1),
    ("店家处理问题很有耐心，态度值得好评", 1),
    ("机器运行很稳定，目前没有发现明显问题", 1),
    ("手机拍照效果很好，续航也比上一代强", 1),
    ("外卖送到的时候还是热的，味道也在线", 1),
    ("这本书内容扎实，对我写论文挺有帮助", 1),
    ("界面虽然简单，但功能很实用", 1),
    ("价格实惠，质量也没有让我失望", 1),
    ("第一次使用感觉不错，学习成本不高", 1),
    ("客服帮我换货很快，整体处理很满意", 1),
    ("电影前半段一般，但后半段非常精彩", 1),
    ("房间不算大，不过卫生和服务都很好", 1),
    ("这门课难度适中，适合我现在的水平", 1),
    ("软件更新后流畅了很多，卡顿明显减少", 1),

    # 负面 0
    ("服务太差了，再也不想去了", 0),
    ("这个产品质量很差，用了两天就坏了", 0),
    ("物流太慢，等了很久才收到", 0),
    ("包装破损严重，里面的东西也被压坏了", 0),
    ("房间隔音很差，晚上根本睡不好", 0),
    ("客服一直敷衍，没有真正解决问题", 0),
    ("电影剧情很无聊，看到一半就想走", 0),
    ("老师讲得太快，很多地方完全没听懂", 0),
    ("课程内容很散，感觉没有体系", 0),
    ("APP经常闪退，体验很糟糕", 0),
    ("耳机戴久了很不舒服，声音也一般", 0),
    ("衣服色差明显，和图片差很多", 0),
    ("价格不便宜，但质量完全不值", 0),
    ("菜品分量少，味道也很普通", 0),
    ("酒店卫生一般，床单上还有污渍", 0),
    ("手机发热严重，电量掉得很快", 0),
    ("页面设计很乱，找一个功能要找半天", 0),
    ("买之前说得很好，出问题后没人管", 0),
    ("虽然包装看起来不错，但实际质量很差", 0),
    ("外卖送到已经凉了，口感很差", 0),
    ("这本书标题很吸引人，但内容很水", 0),
    ("客服回复慢，而且答非所问", 0),
    ("系统更新后反而更卡了", 0),
    ("电影演员不错，但剧情实在太尴尬", 0),
    ("房间位置不错，可是卫生太差", 0),
    ("课程宣传说适合零基础，实际讲得很跳跃", 0),
    ("产品刚开始还行，没多久问题就出来了", 0),
    ("说是当天发货，结果拖了三天", 0),
    ("味道不难吃，但这个价格真的不值得", 0),
    ("售后流程太麻烦，体验很差", 0),
]


def stratified_split(df, valid_ratio=0.2, seed=42):
    train_parts = []
    valid_parts = []

    for label, group in df.groupby("label"):
        group = group.sample(frac=1, random_state=seed).reset_index(drop=True)

        n_valid = max(1, int(len(group) * valid_ratio))

        valid_parts.append(group.iloc[:n_valid])
        train_parts.append(group.iloc[n_valid:])

    train_df = pd.concat(train_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    valid_df = pd.concat(valid_parts).sample(frac=1, random_state=seed).reset_index(drop=True)

    return train_df, valid_df


def main():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    df = pd.DataFrame(samples, columns=["text", "label"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_df, valid_df = stratified_split(df, valid_ratio=0.2, seed=42)

    df.to_csv(data_dir / "all.csv", index=False, encoding="utf-8-sig")
    train_df.to_csv(data_dir / "train.csv", index=False, encoding="utf-8-sig")
    valid_df.to_csv(data_dir / "valid.csv", index=False, encoding="utf-8-sig")

    print("all.csv:", len(df))
    print("train.csv:", len(train_df))
    print("valid.csv:", len(valid_df))
    print("\nlabel distribution in train:")
    print(train_df["label"].value_counts())
    print("\nlabel distribution in valid:")
    print(valid_df["label"].value_counts())


if __name__ == "__main__":
    main()