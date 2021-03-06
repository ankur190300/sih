const Job = require('../models/jobs')

module.exports.create = function(req, res){
    return res.render('job',
            {
                title: 'Job',
                
            });
};

module.exports.create_job = async function(req, res){

    try{

        let p= await Job.create({

            description: req.body.description,
            company: req.user._id,
            job_name: req.body.job_name, 
            salary: req.body.salary

         })

         req.flash('success', "Job created!")
        return res.redirect('back');

    }catch(err){
        req.flash('error', err);
        
        return res.redirect('back');
    }

}

module.exports.delete = async function(req ,res){

    try{

        let job = await Job.findById(req.params.id);
        job.remove();
        req.flash('success', 'Comment deleted!')
        res.redirect('back');


    }catch(err){
        req.flash('error', err);
        
        return res.redirect('back');
    }
    
    

}

module.exports.get_job = function(req, res){

    console.log("hello");


}