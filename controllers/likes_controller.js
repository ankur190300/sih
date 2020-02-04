const Like = require('../models/like');
const Post = require('../models/posts');
const Comment = require('../models/comment');




module.exports.togglelike = async function (req, res) {

    try {
        let likable;
        let deleted = false;

        if (req.query.type == 'Post') {
            likable = await Post.findById(req.query.id).populate('likes');

        } else {
            likable = await Comment.findById(req.query.id).populate('likes');
        }
        

        let existingLike = await Like.findOne({
            user: req.user._id,
            likable: req.query.id,
            onModel:req.query.type
        })
        // if like already exists then delete it 
        if (existingLike) {
            
            likable.likes.pull(existingLike._id);
            likable.save()

            existingLike.remove();
            deleted = true;
        } else {
            // otherwise create a new like
            let newLike = await Like.create({
                user: req.user._id,
                likable: req.query.id,
                onModel: req.query.type,
            });
            likable.likes.push(newLike._id);
            likable.save();
        }

        return res.json(200, {
            message: "request successful",
            data: {
                deleted: deleted
            }
        });
    } catch(err){
        console.log(err);
        return res.json(500, {
            message: 'Internal server error'
        });
    }
}