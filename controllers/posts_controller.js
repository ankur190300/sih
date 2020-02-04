const Post = require('../models/posts');
//const passport = require('')
const Comment = require('../models/comment');
const Like = require('../models/like');
module.exports.create =  async function (req, res) {

    try {
         let p= await Post.create({

            content: req.body.content,
            user: req.user._id
         })

        

        if (req.xhr) {
            
            pos = await p.populate('user', 'name').execPopulate();
            //console.log(pos);
            
            return res.status(200).json({
                data: {
                    post: pos,
                }, 
                message:'post created'
            })
        }

        req.flash('success', "Post published!")
        return res.redirect('back');
        

    } catch(err){
        req.flash('error', err);
        
        return res.redirect('back');
    }
};

module.exports.delete = async function (req, res) {
    
    try {
        let p =await Post.findById(req.params.id);


        //id instead of _id because id returns string
        if (p.user == req.user.id) {


            await Like.deleteMany({ likeable: post, onModel: 'Post' });
            await Like.deleteMany({ _id: { $in: post.comments } });

            p.remove();
            await Comment.deleteMany({ post: req.params.id })
            
            if (req.xhr) {

                return res.status(200).json({
                    data: {
                        post_id: req.params.id,
                    },
                    message: 'post deleted'
                });
            }


            req.flash('success', 'The post and the associated comments deleted');
            return res.redirect('back'); 
        }
        else {
            req.flash('error', 'You are unauthorized to delete this post');
            return res.redirect('back');
        }

        



    }
    catch(err){
        req.flash('error', err);
        console.log('#######', err);
        return res.redirect('back')  

    }


}