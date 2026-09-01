from sqlalchemy import String, func


# md5 over the id and the seed: unlike random() it survives a rebuild of the set
def by_id_and_seed(column, seed):
    return func.md5(func.concat(func.cast(column, String), seed))
