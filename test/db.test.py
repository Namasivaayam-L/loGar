import pymongo

client = pymongo.MongoClient('localhost',27017)
db = client["local"]
collection = db["startup_log"]

# Insert document
# data = {"name": "John Doe", "age": 30}
# result = collection.insert_one(data)
# print(result.inserted_id)

# Find all documents
documents = collection.find()
for doc in documents:
    print(doc)

# # Update document
# query = {"name": "John Doe"}
# newvalues = {"$set": {"age": 31}}
# collection.update_one(query, newvalues)

# # Delete document
# query = {"name": "John Doe"}
# collection.delete_one(query)
