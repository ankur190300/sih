const Post = require('../../../models/posts');
const Comment = require('../../../models/comment');
module.exports.index = async function (req, res) {


    let posts = await Post.find({})
        .sort('-createdAt')
        .populate('user')
        .populate({
            path: 'comments',
            populate: {
                path: 'user'
            }
        })

    return res.json(200, {

        message: "List of Posts",
        posts:posts
    })
};



module.exports.delete = async function (req, res) {

    try {
        let p = await Post.findById(req.params.id);


        //id instead of _id because id returns string
        if (p.user == req.user.id) {

        p.remove();
        await Comment.deleteMany({ post: req.params.id })

        return res.json(200, {

            message: "The post and the associated comments deleted successfully"
        });

            
        }
        else {
            res.json(401, {
                message:"You are unauthorised to delete this post"
            })
        }





    }
    catch (err) {
        console.log(err);
        return res.json(500, {
            message:"Internal server error"
        })

    }


}